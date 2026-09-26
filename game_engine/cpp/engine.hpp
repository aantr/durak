#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace durak {
// Canonical cards: rank * 4 + suit, ranks 6..A, suits C D H S.
int parse_card(const std::string& text);
std::string card_name(int card);
int parse_suit(const std::string& text);
bool beats(int defense, int attack, int trump);

struct Pair { int attack; int defense = -1; };
enum class Kind { Attack, Defend, Take, Pass };
struct Move {
    Kind kind;
    int card = -1;
    int target = -1;  // table pair index, not a card/class id
    bool operator==(const Move& other) const;
    bool operator<(const Move& other) const;
};
std::string move_name(Kind kind);
Kind parse_kind(const std::string& text);

struct State {
    std::array<std::vector<int>, 2> hands;
    std::vector<int> deck;  // front is drawn first, visible trump is last
    std::vector<int> discard;
    std::vector<Pair> table;
    int trump = 0;
    int attacker = 0;
    int turn = 0;
    int attack_limit = 6;
    bool taking = false;
    bool attacker_passed = false;
    int winner = -1;
    bool simultaneous_attacker_wins = true;

    void validate() const;
    std::vector<Move> legal_moves() const;
    void apply(const Move& move);
    void apply_legal(const Move& move);
    void finish_round();
    int uncovered() const;
};
State new_game(std::uint64_t seed, int first_attacker = -1,
               bool simultaneous_attacker_wins = true);

struct Observation {
    std::vector<int> hand, known_opponent, discard;
    std::vector<Pair> table;
    int opponent_count = 0;
    int deck_count = 0;
    int trump = 0;
    int bottom_trump = -1;
    int attacker = 0;
    int turn = 0;
    int attack_limit = 6;
    bool taking = false;
    bool attacker_passed = false;
    bool simultaneous_attacker_wins = true;
    void validate() const;
    State sample(std::mt19937_64& rng) const;
};
struct MoveStats { Move move; int visits = 0; double value = 0; };
struct SearchResult {
    Move best{Kind::Pass};
    std::vector<MoveStats> moves;
    int iterations = 0;
    int terminal_rollouts = 0;
    int deals_started = 0;
    int deals_completed = 0;
    int threads = 1;
    bool determinized = false;
    double elapsed_ms = 0;
};
SearchResult search(const Observation& observation, int iterations,
                    double time_limit_ms, std::uint64_t seed,
                    int rollout_depth = 256, double exploration = 1.41421356237,
                    int rollouts = 0, int deals = 1, int threads = 1);
}  // namespace durak
