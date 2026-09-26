#include "engine.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace durak {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument(message);
}
void add_cards(std::array<bool, 36>& seen, const std::vector<int>& cards) {
    for (int card : cards) {
        require(card >= 0 && card < 36, "Card outside 36-card deck");
        require(!seen[card], "Duplicate card: " + card_name(card));
        seen[card] = true;
    }
}
std::vector<int> table_cards(const std::vector<Pair>& table) {
    std::vector<int> cards;
    for (auto pair : table) {
        cards.push_back(pair.attack);
        if (pair.defense != -1) cards.push_back(pair.defense);
    }
    return cards;
}
void erase_card(std::vector<int>& cards, int card) {
    auto pos = std::find(cards.begin(), cards.end(), card);
    if (pos == cards.end()) throw std::logic_error("Played card is not in hand");
    cards.erase(pos);
}
double heuristic(const State& state) {
    // A bounded evaluation at a truncated rollout, NOT a measured win probability.
    const double difference = double(state.hands[1].size()) - double(state.hands[0].size());
    return 0.5 + 0.4 * std::tanh(difference / 6.0);
}
Move rollout_move(const State& state, const std::vector<Move>& legal, std::mt19937_64& rng) {
    if (rng() % 5 == 0) return legal[rng() % legal.size()];
    // Mostly use cheap non-trumps; retain random exploration, including taking/pass.
    double best_cost = 1e9;
    Move best = legal.front();
    for (auto move : legal) {
        double cost = move.kind == Kind::Take ? 30.0 : 24.0;
        if (move.card >= 0)
            cost = move.card / 4 + (move.card % 4 == state.trump ? 12.0 : 0.0);
        cost += double(rng() % 1000) / 500.0;
        if (cost < best_cost) { best_cost = cost; best = move; }
    }
    return best;
}
struct Node {
    int visits = 0;
    int availability = 0;
    double total = 0;  // all rewards are from player 0's perspective
    std::map<Move, std::unique_ptr<Node>> children;
};
}  // namespace

int parse_suit(const std::string& text) {
    auto pos = std::string("CDHS").find(text);
    require(text.size() == 1 && pos != std::string::npos, "Suit must be C, D, H or S");
    return int(pos);
}
int parse_card(const std::string& text) {
    const std::array<std::string, 9> ranks = {"6","7","8","9","10","J","Q","K","A"};
    require(text.size() >= 2, "Invalid card: " + text);
    int suit = parse_suit(text.substr(text.size() - 1));
    auto rank = std::find(ranks.begin(), ranks.end(), text.substr(0, text.size() - 1));
    require(rank != ranks.end(), "Invalid rank: " + text);
    return int(rank - ranks.begin()) * 4 + suit;
}
std::string card_name(int card) {
    require(card >= 0 && card < 36, "Invalid card id");
    const std::array<std::string, 9> ranks = {"6","7","8","9","10","J","Q","K","A"};
    return ranks[card / 4] + "CDHS"[card % 4];
}
bool beats(int defense, int attack, int trump) {
    return (defense % 4 == attack % 4 && defense / 4 > attack / 4)
        || (defense % 4 == trump && attack % 4 != trump);
}
bool Move::operator==(const Move& other) const {
    return kind == other.kind && card == other.card && target == other.target;
}
bool Move::operator<(const Move& other) const {
    return std::tie(kind, card, target) < std::tie(other.kind, other.card, other.target);
}
std::string move_name(Kind kind) {
    switch (kind) {
        case Kind::Attack: return "attack";
        case Kind::Defend: return "defend";
        case Kind::Take: return "take";
        case Kind::Pass: return "pass";
    }
    throw std::logic_error("Invalid move type");
}
Kind parse_kind(const std::string& text) {
    for (Kind kind : {Kind::Attack, Kind::Defend, Kind::Take, Kind::Pass})
        if (move_name(kind) == text) return kind;
    throw std::invalid_argument("Unknown move type: " + text);
}
int State::uncovered() const {
    return int(std::count_if(table.begin(), table.end(), [](Pair p) { return p.defense == -1; }));
}
void State::validate() const {
    require(trump >= 0 && trump < 4, "Invalid trump suit");
    require((attacker == 0 || attacker == 1) && (turn == 0 || turn == 1), "Invalid player");
    require(winner >= -1 && winner <= 1, "Invalid winner");
    require(attack_limit >= 0 && attack_limit <= 6, "Attack limit must be 0..6");
    require(table.size() <= std::size_t(attack_limit), "Too many attack cards on table");
    require(!(taking && attacker_passed), "A passed take must already be resolved");
    std::array<bool, 36> seen{};
    add_cards(seen, hands[0]); add_cards(seen, hands[1]);
    add_cards(seen, deck); add_cards(seen, discard); add_cards(seen, table_cards(table));
    require(std::count(seen.begin(), seen.end(), true) == 36, "State must account for all 36 cards");
    for (auto pair : table)
        require(pair.defense == -1 || beats(pair.defense, pair.attack, trump), "Defense does not beat attack");
    if (winner == -1) {
        int defender_start = int(hands[1-attacker].size()) + int(table.size()) - uncovered();
        require(attack_limit <= std::min(6, defender_start), "Attack limit exceeds defender's initial hand");
        require(!(deck.empty() && table.empty() && (hands[0].empty() || hands[1].empty())),
                "Finished position must specify winner");
        require(!table.empty() || (turn == attacker && !taking && !attacker_passed), "Invalid empty table phase");
        require(!taking || turn == attacker, "After take it is the attacker's turn");
        require(!attacker_passed || (turn != attacker && uncovered() > 0), "Invalid attacker_passed phase");
        require(turn == attacker || uncovered() > 0, "Defender has nothing to beat");
        require(taking || uncovered() <= int(hands[1-attacker].size()), "Defender cannot cover this many attacks; mark taking");
        require(!legal_moves().empty(), "Active position has no legal moves");
    } else {
        require(deck.empty() && table.empty() && hands[winner].empty(), "Invalid finished position");
    }
}
std::vector<Move> State::legal_moves() const {
    std::vector<Move> moves;
    if (winner != -1) return moves;
    if (turn == attacker) {
        std::array<bool, 9> ranks{};
        for (auto pair : table) {
            ranks[pair.attack / 4] = true;
            if (pair.defense >= 0) ranks[pair.defense / 4] = true;
        }
        if (!attacker_passed && table.size() < std::size_t(attack_limit))
            for (int card : hands[turn])
                if (table.empty() || ranks[card / 4]) moves.push_back({Kind::Attack, card});
        if (!table.empty()) moves.push_back({Kind::Pass});
    } else {
        for (std::size_t i = 0; i < table.size(); ++i)
            if (table[i].defense < 0)
                for (int card : hands[turn])
                    if (beats(card, table[i].attack, trump)) moves.push_back({Kind::Defend, card, int(i)});
        if (uncovered() > 0) moves.push_back({Kind::Take});
    }
    return moves;
}
void State::apply(const Move& move) {
    auto legal = legal_moves();
    require(std::find(legal.begin(), legal.end(), move) != legal.end(), "Illegal move");
    apply_legal(move);
}
void State::apply_legal(const Move& move) {
    switch (move.kind) {
        case Kind::Attack:
            erase_card(hands[attacker], move.card);
            table.push_back({move.card});
            if (!taking) turn = 1 - attacker;
            break;
        case Kind::Defend:
            erase_card(hands[1-attacker], move.card);
            table[move.target].defense = move.card;
            if (uncovered() == 0) {
                if (attacker_passed) finish_round();
                else turn = attacker;
            }
            break;
        case Kind::Take:
            taking = true;
            if (attacker_passed) finish_round();
            else turn = attacker;
            break;
        case Kind::Pass:
            attacker_passed = true;
            if (taking || uncovered() == 0) finish_round();
            else turn = 1 - attacker;
            break;
    }
}
void State::finish_round() {
    int old_attacker = attacker;
    int defender = 1 - attacker;
    auto cards = table_cards(table);
    auto& destination = taking ? hands[defender] : discard;
    destination.insert(destination.end(), cards.begin(), cards.end());
    table.clear();
    // Draw order matters, especially for the last face-up trump.
    for (int player : {old_attacker, defender})
        while (hands[player].size() < 6 && !deck.empty()) {
            hands[player].push_back(deck.front());
            deck.erase(deck.begin());
        }
    if (deck.empty() && (hands[0].empty() || hands[1].empty())) {
        if (hands[0].empty() && hands[1].empty())
            winner = simultaneous_attacker_wins ? old_attacker : defender;
        else winner = hands[0].empty() ? 0 : 1;
    }
    attacker = taking ? old_attacker : defender;
    turn = attacker;
    taking = attacker_passed = false;
    attack_limit = std::min(6, int(hands[1-attacker].size()));
}
State new_game(std::uint64_t seed, int first_attacker, bool simultaneous_attacker_wins) {
    require(first_attacker >= -1 && first_attacker <= 1, "Invalid first attacker");
    State state;
    state.simultaneous_attacker_wins = simultaneous_attacker_wins;
    std::mt19937_64 rng(seed);
    state.deck.resize(36);
    std::iota(state.deck.begin(), state.deck.end(), 0);
    std::shuffle(state.deck.begin(), state.deck.end(), rng);
    state.trump = state.deck.back() % 4;
    for (int i = 0; i < 12; ++i) state.hands[i % 2].push_back(state.deck[i]);
    state.deck.erase(state.deck.begin(), state.deck.begin() + 12);
    if (first_attacker < 0) {
        int lowest = 36;
        for (int p = 0; p < 2; ++p)
            for (int card : state.hands[p])
                if (card % 4 == state.trump && card < lowest) { lowest = card; state.attacker = p; }
    } else state.attacker = first_attacker;
    state.turn = state.attacker;
    state.validate();
    return state;
}
void Observation::validate() const {
    require(opponent_count >= 0 && opponent_count <= 36 && deck_count >= 0 && deck_count <= 36, "Invalid hidden card counts");
    require(int(known_opponent.size()) <= opponent_count, "More known opponent cards than opponent count");
    require(int(hand.size() + table_cards(table).size() + discard.size()) + opponent_count + deck_count == 36,
            "Observation counts do not sum to 36; check OCR/discard history");
    if (bottom_trump >= 0) {
        require(bottom_trump < 36 && deck_count > 0, "Bottom trump requires a nonempty deck");
        require(bottom_trump % 4 == trump, "Bottom card does not match trump suit");
    }
    std::array<bool, 36> seen{};
    add_cards(seen, hand); add_cards(seen, known_opponent);
    add_cards(seen, discard); add_cards(seen, table_cards(table));
    if (bottom_trump >= 0) add_cards(seen, {bottom_trump});
    std::mt19937_64 rng(0);
    sample(rng).validate();
}
State Observation::sample(std::mt19937_64& rng) const {
    std::array<bool, 36> seen{};
    add_cards(seen, hand); add_cards(seen, known_opponent);
    add_cards(seen, discard); add_cards(seen, table_cards(table));
    if (bottom_trump >= 0) add_cards(seen, {bottom_trump});
    std::vector<int> pool;
    for (int card = 0; card < 36; ++card) if (!seen[card]) pool.push_back(card);
    std::shuffle(pool.begin(), pool.end(), rng);
    int bottom = bottom_trump;
    if (deck_count > 0 && bottom < 0) {
        // Even if its rank wasn't read, the last face-up card is a trump.
        auto pos = std::find_if(pool.begin(), pool.end(), [&](int card) { return card % 4 == trump; });
        require(pos != pool.end(), "No possible bottom trump remains in the hidden pool");
        bottom = *pos;
        pool.erase(pos);
    }
    int missing = opponent_count - int(known_opponent.size());
    require(missing >= 0 && missing <= int(pool.size()), "Impossible hidden hand size");
    State state;
    state.hands = {hand, known_opponent};
    state.hands[1].insert(state.hands[1].end(), pool.begin(), pool.begin() + missing);
    state.deck.assign(pool.begin() + missing, pool.end());
    if (bottom >= 0) state.deck.push_back(bottom);
    require(int(state.deck.size()) == deck_count, "Hidden deck size mismatch");
    state.discard = discard; state.table = table;
    state.trump = trump; state.attacker = attacker; state.turn = turn;
    state.attack_limit = attack_limit; state.taking = taking;
    state.attacker_passed = attacker_passed;
    state.simultaneous_attacker_wins = simultaneous_attacker_wins;
    return state;
}
namespace {
using Clock = std::chrono::steady_clock;

// Each tree and RNG belongs to a single task; no shared mutable tree nodes.
SearchResult search_tree(const Observation& obs, int iterations, Clock::time_point deadline,
                         std::uint64_t seed, int rollout_depth, double exploration,
                         bool determinized, bool guarantee_one) {
    std::mt19937_64 rng(seed);
    State fixed_world;
    if (determinized) fixed_world = obs.sample(rng);
    Node root;
    SearchResult result;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if ((iteration > 0 || !guarantee_one) && Clock::now() >= deadline) break;
        State state = determinized ? fixed_world : obs.sample(rng);
        Node* node = &root;
        std::vector<Node*> path{node};
        int depth = 0;
        for (; state.winner == -1 && depth < rollout_depth; ++depth) {
            auto legal = state.legal_moves();
            if (legal.empty()) throw std::logic_error("MCTS reached invalid state");
            std::vector<Move> unvisited;
            for (auto move : legal) {
                auto& child = node->children[move];
                if (!child) child = std::make_unique<Node>();
                ++child->availability;
                if (!child->visits) unvisited.push_back(move);
            }
            bool expand = !unvisited.empty();
            Move selected = legal.front();
            if (expand) selected = unvisited[rng() % unvisited.size()];
            else {
                double best_score = -1e9;
                for (auto move : legal) {
                    auto& child = *node->children.at(move);
                    double mean = child.total / child.visits;
                    double utility = state.turn == 0 ? mean : 1.0 - mean;
                    double score = utility + exploration * std::sqrt(std::log(double(child.availability)) / child.visits);
                    if (score > best_score) { best_score = score; selected = move; }
                }
            }
            state.apply_legal(selected);
            node = node->children.at(selected).get();
            path.push_back(node);
            if (expand) { ++depth; break; }
        }
        for (; state.winner == -1 && depth < rollout_depth; ++depth) {
            auto legal = state.legal_moves();
            if (legal.empty()) throw std::logic_error("Rollout reached invalid state");
            state.apply_legal(rollout_move(state, legal, rng));
        }
        double reward = state.winner == -1 ? heuristic(state) : (state.winner == 0 ? 1.0 : 0.0);
        if (state.winner != -1) ++result.terminal_rollouts;
        for (auto* visited : path) { ++visited->visits; visited->total += reward; }
        ++result.iterations;
    }
    for (auto& entry : root.children) {
        const auto& child = *entry.second;
        result.moves.push_back({entry.first, child.visits, child.visits ? child.total / child.visits : 0.0});
    }
    return result;
}

std::uint64_t task_seed(std::uint64_t seed, int task) {
    if (task == 0) return seed;  // preserve old single-thread ISMCTS sequence
    auto z = seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(task);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
}  // namespace

SearchResult search(const Observation& obs, int iterations, double time_limit_ms,
                    std::uint64_t seed, int rollout_depth, double exploration,
                    int rollouts, int deals, int threads) {
    require(iterations > 0 && iterations <= 10000000, "iterations must be 1..10000000");
    require(std::isfinite(time_limit_ms) && time_limit_ms >= 0, "Invalid time limit");
    require(rollout_depth > 0 && rollout_depth <= 10000, "rollout_depth must be 1..10000");
    require(std::isfinite(exploration) && exploration >= 0, "Invalid exploration constant");
    require(rollouts >= 0 && rollouts <= 10000000, "rollouts must be 0..10000000 (0 = ISMCTS)");
    require(deals > 0 && deals <= 100000, "deals must be 1..100000");
    require(rollouts > 0 || deals == 1, "deals requires rollouts");
    require(static_cast<std::int64_t>(rollouts) * deals <= 10000000,
            "rollouts * deals must not exceed 10000000");
    require(threads > 0 && threads <= 256, "threads must be 1..256");
    const auto start = Clock::now();
    auto deadline = Clock::time_point::max();
    // Avoid overflowing the clock when passed a very large but finite limit.
    const double available_ms = std::chrono::duration<double, std::milli>(deadline - start).count();
    if (time_limit_ms > 0 && time_limit_ms < available_ms)
        deadline = start + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double, std::milli>(time_limit_ms));
    obs.validate();
    require(obs.turn == 0, "Recommendations require player 0's turn");
    const bool determinized = rollouts > 0;
    const int tasks = determinized ? deals : std::min(threads, iterations);
    const int workers = std::min(threads, tasks);
    std::vector<SearchResult> partial(tasks);
    auto work = [&](int worker) {
        for (int task = worker; task < tasks; task += workers) {
            if (task != 0 && Clock::now() >= deadline) break;
            const int budget = determinized ? rollouts : iterations / tasks + (task < iterations % tasks);
            partial[task] = search_tree(obs, budget, deadline, task_seed(seed, task),
                                        rollout_depth, exploration, determinized, task == 0);
            if (determinized && partial[task].iterations > 0) {
                partial[task].deals_started = 1;
                partial[task].deals_completed = partial[task].iterations == budget ? 1 : 0;
            }
        }
    };
    std::vector<std::future<void>> futures;
    // std::future joins even during exception unwinding; worker exceptions reach Python.
    for (int worker = 1; worker < workers; ++worker)
        futures.push_back(std::async(std::launch::async, work, worker));
    work(0);
    for (auto& future : futures) future.get();

    SearchResult result;
    result.determinized = determinized;
    result.threads = workers;
    std::map<Move, std::pair<int, double>> totals;
    // Stable reduction order makes fixed-budget determinized searches independent of thread count.
    for (const auto& part : partial) {
        result.iterations += part.iterations;
        result.terminal_rollouts += part.terminal_rollouts;
        result.deals_started += part.deals_started;
        result.deals_completed += part.deals_completed;
        for (const auto& stat : part.moves) {
            auto& total = totals[stat.move];
            total.first += stat.visits;
            total.second += stat.value * stat.visits;
        }
    }
    for (const auto& entry : totals) {
        const auto& total = entry.second;
        result.moves.push_back({entry.first, total.first, total.first ? total.second / total.first : 0.0});
    }
    std::sort(result.moves.begin(), result.moves.end(), [determinized](const MoveStats& a, const MoveStats& b) {
        if ((a.visits > 0) != (b.visits > 0)) return a.visits > 0;
        // Match the reference's W/N choice for determinizations; retain robust-child ISMCTS.
        if (determinized && a.value != b.value) return a.value > b.value;
        if (a.visits != b.visits) return a.visits > b.visits;
        if (a.value != b.value) return a.value > b.value;
        return a.move < b.move;
    });
    require(!result.moves.empty(), "No move available");
    result.best = result.moves.front().move;
    result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return result;
}
}  // namespace durak
