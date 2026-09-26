import unittest

from game_state.game import _card_code, _normalise_suit


class SuitNormalizationTests(unittest.TestCase):
    def test_classifier_names_and_aliases(self):
        for expected, names in (
            ("C", ("clubs", "club", "kresti", " C l u b s ")),
            ("S", ("spades", "spade", "piki", "SPADES")),
            ("D", ("diamonds", "diamond", "bubi")),
            ("H", ("hearts", "heart")),
        ):
            for name in names:
                with self.subTest(name=name):
                    self.assertEqual(_normalise_suit(name), expected)
                    self.assertEqual(_card_code(("8B", name)), "8" + expected)

    def test_canonical_codes_and_symbols_unchanged(self):
        for suit, symbol in (("C", "♣"), ("S", "♠"), ("D", "♦"), ("H", "♥")):
            with self.subTest(suit=suit):
                self.assertEqual(_normalise_suit(suit), suit)
                self.assertEqual(_card_code("8" + suit), "8" + suit)
                self.assertEqual(_card_code(("8", symbol)), "8" + suit)


if __name__ == "__main__":
    unittest.main()
