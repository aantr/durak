#include "engine.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace durak;
namespace {
template<class T> T get(const py::dict& d, const char* key, T fallback) {
    return d.contains(key) && !d[key].is_none() ? py::cast<T>(d[key]) : fallback;
}
std::vector<int> cards(py::handle values) {
    std::vector<int> result;
    for (auto value : py::reinterpret_borrow<py::iterable>(values))
        result.push_back(parse_card(py::cast<std::string>(value)));
    return result;
}
std::vector<int> card_list(const py::dict& d, const char* key) {
    return d.contains(key) ? cards(d[key]) : std::vector<int>{};
}
py::list names(const std::vector<int>& values) {
    py::list result;
    for (int value : values) result.append(card_name(value));
    return result;
}
std::vector<Pair> table(const py::dict& d) {
    std::vector<Pair> result;
    if (d.contains("table")) for (auto item : py::cast<py::list>(d["table"])) {
        auto pair = py::cast<py::dict>(item);
        result.push_back({parse_card(py::cast<std::string>(pair["attack"])),
            pair.contains("defense") && !pair["defense"].is_none()
                ? parse_card(py::cast<std::string>(pair["defense"])) : -1});
    }
    return result;
}
bool tie_rule(const py::dict& d) {
    auto rule = get<std::string>(d, "simultaneous_winner", "attacker");
    if (rule != "attacker" && rule != "defender") throw std::invalid_argument("Invalid simultaneous_winner");
    return rule == "attacker";
}
template<class T> void common(T& s, const py::dict& d) {
    s.table = table(d);
    s.discard = card_list(d, "discard");
    s.trump = parse_suit(py::cast<std::string>(d["trump"]));
    s.attacker = get<int>(d, "attacker", 0);
    s.turn = get<int>(d, "turn", s.attacker);
    s.attack_limit = get<int>(d, "attack_limit", 6);
    s.taking = get<bool>(d, "taking", false);
    s.attacker_passed = get<bool>(d, "attacker_passed", false);
    s.simultaneous_attacker_wins = tie_rule(d);
}
State read_state(const py::dict& d) {
    State s;
    common(s, d);
    auto hands = py::cast<py::list>(d["hands"]);
    if (hands.size() != 2) throw std::invalid_argument("Exactly two players required");
    s.hands = {cards(hands[0]), cards(hands[1])};
    s.deck = card_list(d, "deck");
    s.winner = get<int>(d, "winner", -1);
    s.validate();
    return s;
}
Observation read_observation(const py::dict& d) {
    Observation o;
    common(o, d);
    o.hand = card_list(d, "hand");
    o.known_opponent = card_list(d, "known_opponent");
    o.opponent_count = py::cast<int>(d["opponent_count"]);
    o.deck_count = py::cast<int>(d["deck_count"]);
    if (d.contains("bottom_trump") && !d["bottom_trump"].is_none())
        o.bottom_trump = parse_card(py::cast<std::string>(d["bottom_trump"]));
    return o;
}
py::dict move_dict(const Move& move) {
    py::dict result;
    result["type"] = move_name(move.kind);
    result["card"] = move.card < 0 ? py::none() : py::cast(card_name(move.card));
    result["target"] = move.target < 0 ? py::none() : py::cast(move.target);
    return result;
}
Move read_move(const py::dict& d) {
    return {parse_kind(py::cast<std::string>(d["type"])),
        d.contains("card") && !d["card"].is_none() ? parse_card(py::cast<std::string>(d["card"])) : -1,
        get<int>(d, "target", -1)};
}
py::list move_list(const State& s) {
    py::list result;
    for (auto move : s.legal_moves()) result.append(move_dict(move));
    return result;
}
py::dict state_dict(const State& s) {
    py::dict d;
    py::list hands;
    hands.append(names(s.hands[0])); hands.append(names(s.hands[1]));
    d["hands"] = hands; d["deck"] = names(s.deck); d["discard"] = names(s.discard);
    py::list pairs;
    for (auto pair : s.table) {
        py::dict p;
        p["attack"] = card_name(pair.attack);
        p["defense"] = pair.defense < 0 ? py::none() : py::cast(card_name(pair.defense));
        pairs.append(p);
    }
    d["table"] = pairs; d["trump"] = std::string(1, "CDHS"[s.trump]);
    d["attacker"] = s.attacker; d["turn"] = s.turn; d["attack_limit"] = s.attack_limit;
    d["taking"] = s.taking; d["attacker_passed"] = s.attacker_passed; d["winner"] = s.winner;
    d["simultaneous_winner"] = s.simultaneous_attacker_wins ? "attacker" : "defender";
    return d;
}
py::dict result_dict(const SearchResult& s) {
    py::dict result;
    result["action"] = move_dict(s.best);
    result["iterations"] = s.iterations;
    result["terminal_rollouts"] = s.terminal_rollouts;
    result["cutoff_rollouts"] = s.iterations - s.terminal_rollouts;
    result["elapsed_ms"] = s.elapsed_ms;
    py::list moves;
    for (const auto& stat : s.moves) {
        auto d = move_dict(stat.move);
        d["visits"] = stat.visits;
        d["value"] = stat.visits ? py::cast(stat.value) : py::none();
        moves.append(d);
    }
    result["moves"] = moves;
    return result;
}
}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Two-player 36-card Durak and single-observer information-set MCTS";
    py::class_<State>(m, "Game")
        .def(py::init(&read_state), py::arg("state"))
        .def_static("new_game", &new_game, py::arg("seed"), py::arg("attacker") = -1,
                    py::arg("simultaneous_attacker_wins") = true)
        .def("snapshot", &state_dict)
        .def("legal_moves", &move_list)
        .def("apply", [](State& s, const py::dict& move) { s.apply(read_move(move)); })
        .def("validate", &State::validate)
        .def_property_readonly("winner", [](const State& s) { return s.winner; });
    m.def("analyze", [](const py::dict& observation, int iterations, double time_limit_ms,
                         std::uint64_t seed, int rollout_depth, double exploration) {
        auto obs = read_observation(observation);
        SearchResult result;
        { py::gil_scoped_release release;
          result = search(obs, iterations, time_limit_ms, seed, rollout_depth, exploration); }
        return result_dict(result);
    }, py::arg("observation"), py::arg("iterations") = 3000, py::arg("time_limit_ms") = 250,
       py::arg("seed") = 0, py::arg("rollout_depth") = 256, py::arg("exploration") = 1.41421356237);
    m.def("sample_world", [](const py::dict& d, std::uint64_t seed) {
        auto obs = read_observation(d); obs.validate();
        std::mt19937_64 rng(seed);
        return state_dict(obs.sample(rng));
    }, py::arg("observation"), py::arg("seed") = 0);
}
