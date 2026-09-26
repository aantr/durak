import copy
import random
import threading
import unittest
from types import SimpleNamespace

from game_engine import DurakEngine, observation_from_state
from game_engine import _native

CARDS = {rank + suit for rank in ('6','7','8','9','10','J','Q','K','A') for suit in 'CDHS'}


def position(hands, *, table=None, deck=None, **kwargs):
    table, deck = table or [], deck or []
    used = set(hands[0] + hands[1] + deck)
    for pair in table:
        used.add(pair['attack'])
        if pair.get('defense'): used.add(pair['defense'])
    result = dict(hands=hands, table=table, deck=deck, discard=sorted(CARDS-used),
                  trump='S', attacker=0, turn=0, attack_limit=6)
    result.update(kwargs)
    if 'attack_limit' not in kwargs:
        result['attack_limit'] = min(6, len(hands[1-result['attacker']]) + sum(bool(p.get('defense')) for p in table))
    return result


def observation(full):
    return dict(hand=full['hands'][0], known_opponent=full['hands'][1],
                opponent_count=len(full['hands'][1]), deck_count=len(full['deck']),
                discard=full['discard'], table=full['table'], trump=full['trump'],
                attacker=full['attacker'], turn=full['turn'], attack_limit=full['attack_limit'],
                taking=full.get('taking',False), attacker_passed=full.get('attacker_passed',False))


class RulesTests(unittest.TestCase):
    def test_deal_and_face_up_trump(self):
        game = _native.Game.new_game(7)
        s = game.snapshot()
        self.assertEqual(list(map(len,s['hands'])), [6,6])
        self.assertEqual(len(s['deck']),24)
        self.assertTrue(s['deck'][-1].endswith(s['trump']))
        self.assertEqual(s, _native.Game.new_game(7).snapshot())

    def test_defense_suit_rank_trump_and_target(self):
        game = _native.Game(position([['6C','AS'],['7C','6D','6S']],
                                    table=[{'attack':'KC','defense':None}], attacker=1,turn=0))
        moves = game.legal_moves()
        self.assertIn(dict(type='defend',card='AS',target=0),moves)
        self.assertNotIn(dict(type='defend',card='6C',target=0),moves)
        with self.assertRaises(ValueError): game.apply(dict(type='defend',card='AS',target=1))
        game.apply(dict(type='defend',card='AS',target=0))
        game.validate()

    def test_throw_matches_defense_rank(self):
        game = _native.Game(position([['8D','9D','AC'],['KS']],
                                    table=[{'attack':'8H','defense':'9H'}]))
        attacks = {m['card'] for m in game.legal_moves() if m['type']=='attack'}
        self.assertEqual(attacks,{'8D','9D'})

    def test_limit_and_no_pass_on_empty_table(self):
        game = _native.Game(position([['6C'],['7C']],attack_limit=1))
        self.assertEqual([m['type'] for m in game.legal_moves()],['attack'])
        game.apply(dict(type='attack',card='6C'))
        game.apply(dict(type='take'))
        self.assertEqual([m['type'] for m in game.legal_moves()],['pass'])

    def test_take_allows_extra_throws_and_skips_defender(self):
        game = _native.Game(position([['6D','AC'],['7C','7D']],table=[{'attack':'6C'}],
                                    taking=True,attack_limit=2,deck=['8C','9C','10C']))
        game.apply(dict(type='attack',card='6D'))
        self.assertEqual(game.snapshot()['turn'],0)
        game.apply(dict(type='pass'))
        s=game.snapshot()
        self.assertTrue({'6C','6D','7C'} <= set(s['hands'][1]))
        self.assertEqual(s['attacker'],0)
        self.assertFalse(s['table'])

    def test_attacker_pass_then_defender_can_finish(self):
        game = _native.Game(position([['AC'],['7C']],table=[{'attack':'6C'}]))
        game.apply(dict(type='pass'))
        self.assertTrue(game.snapshot()['attacker_passed'])
        game.apply(dict(type='defend',card='7C',target=0))
        self.assertEqual(game.winner,1)
        self.assertTrue({'6C','7C'} <= set(game.snapshot()['discard']))

    def test_refill_attacker_first_bottom_trump_last(self):
        game = _native.Game(position([['6D'],['8C']],table=[{'attack':'6C','defense':'7C'}],
                                    deck=['9C','AS']))
        game.apply(dict(type='pass'))
        s=game.snapshot()
        self.assertEqual(s['hands'][0],['6D','9C','AS'])
        self.assertEqual(s['hands'][1],['8C'])
        self.assertEqual(s['attacker'],1)

    def test_simultaneous_finish_attacker_wins_after_defense(self):
        game = _native.Game(position([['6C'],['7C']],attack_limit=1))
        game.apply(dict(type='attack',card='6C'))
        self.assertEqual(game.winner,-1)  # winner is resolved at round end
        game.apply(dict(type='defend',card='7C',target=0))
        game.apply(dict(type='pass'))
        self.assertEqual(game.winner,0)
        self.assertEqual(game.legal_moves(),[])
        game.validate()

    def test_bad_states_and_moves(self):
        data=position([['6C'],['7C']])
        data['hands'][1].append('6C')
        with self.assertRaises(ValueError): _native.Game(data)
        game=_native.Game(position([['6C'],['7C']]))
        before=game.snapshot()
        with self.assertRaises(ValueError): game.apply(dict(type='attack',card='AS'))
        self.assertEqual(before,game.snapshot())
        with self.assertRaises(ValueError):
            _native.Game(position([['6C'],['7C']],attack_limit=6))

    def test_random_play_card_conservation(self):
        rng=random.Random(11)
        finished=0
        for seed in range(40):
            game=_native.Game.new_game(seed)
            for _ in range(1500):
                if game.winner != -1:
                    finished += 1
                    break
                moves=game.legal_moves()
                self.assertTrue(moves)
                game.apply(rng.choice(moves))
                game.validate()
            self.assertIn(game.winner,(-1,0,1))
        self.assertGreater(finished,30)


class SearchTests(unittest.TestCase):
    def test_hidden_world_conservation_and_known_bottom(self):
        full=_native.Game.new_game(5,attacker=0).snapshot()
        obs=observation(full)
        obs['known_opponent']=full['hands'][1][:2]
        obs['bottom_trump']=full['deck'][-1]
        worlds=[]
        for seed in range(12):
            world=_native.sample_world(obs,seed)
            _native.Game(world).validate()
            self.assertTrue(set(obs['known_opponent']) <= set(world['hands'][1]))
            self.assertEqual(world['deck'][-1],obs['bottom_trump'])
            worlds.append(world['hands'][1])
        self.assertNotEqual(worlds[0],worlds[1])

    def test_search_finds_winning_defense(self):
        full=position([['7C'],['8D']],table=[{'attack':'6C'}],attacker=1,turn=0,attack_limit=1)
        result=_native.analyze(observation(full),iterations=500,time_limit_ms=0,seed=8)
        self.assertEqual(result['action'],dict(type='defend',card='7C',target=0))
        self.assertEqual(sum(m['visits'] for m in result['moves']),500)
        self.assertGreater(result['terminal_rollouts'],0)

    def test_unknown_bottom_still_has_trump_suit(self):
        full=_native.Game.new_game(15,attacker=0).snapshot()
        obs=observation(full)
        obs['known_opponent']=[]
        for seed in range(30):
            world=_native.sample_world(obs,seed)
            self.assertTrue(world['deck'][-1].endswith(obs['trump']))
            _native.Game(world).validate()

    def test_waiting_without_unknown_fields(self):
        engine=DurakEngine('S')
        self.assertEqual(engine.suggest({'phase':'opponent_turn'})['status'],'waiting')
        self.assertEqual(engine.suggest({'phase':'ready'})['status'],'waiting')

    def test_reproducible_search_and_budget(self):
        obs=observation(_native.Game.new_game(2,attacker=0).snapshot())
        obs['known_opponent']=[]
        a=_native.analyze(obs,iterations=100,time_limit_ms=0,seed=3)
        b=_native.analyze(obs,iterations=100,time_limit_ms=0,seed=3)
        self.assertEqual(a['moves'],b['moves'])
        quick=_native.analyze(obs,iterations=100000,time_limit_ms=10,seed=3)
        self.assertGreater(quick['iterations'],0)
        self.assertLess(quick['iterations'],100000)
        self.assertLess(quick['elapsed_ms'],1000)

    def test_adapter_and_recommendation(self):
        full=position([['7C'],['8D']],table=[{'attack':'6C'}],attacker=1,turn=0,attack_limit=1)
        state=SimpleNamespace(phase='defend_or_take',hand_cards={'7C'},field_cards={'6C'},
            field_layout=[{'card':'6C','covers':None}],out_cards=set(full['discard']),
            known_opponent_cards={'8D'},opponent_card_count=1,deck_remaining=0,
            mine_text='',opponent_text='')
        before=copy.deepcopy(vars(state))
        result=DurakEngine('S',iterations=200,time_limit_ms=0,seed=8).suggest(state)
        self.assertEqual(result['action']['target_card'],'6C')
        self.assertEqual(result['action']['card'],'7C')
        self.assertEqual(vars(state),before)

    def test_pass_and_bat_buttons_offer_throw_in_or_finish(self):
        for button, opponent, defense in [('Pass','I take',None),('Bat','something','7C')]:
            with self.subTest(button=button):
                full=position([['6D','AC'],['8D','9D']],table=[{'attack':'6C','defense':defense}])
                layout=[{'card':'6C','covers':None}]
                if defense:
                    layout.append({'card':defense,'covers':0})
                state=dict(phase='opponent_turn',button=button,opponent=opponent,mine='',
                    hand_cards=full['hands'][0],field_cards=[p['card'] for p in layout],field_layout=layout,
                    deck_remaining=0,opponent_card_count=2,known_opponent_cards=full['hands'][1],out_cards=full['discard'])
                obs=observation_from_state(state,trump='S')
                self.assertEqual((obs['attacker'],obs['turn']),(0,0))
                self.assertEqual(obs['taking'],button=='Pass')
                result=DurakEngine('S',iterations=100,time_limit_ms=0,seed=4).suggest(state)
                self.assertEqual(result['status'],'ok')
                self.assertTrue(any(m['type']=='attack' and m['card']=='6D' for m in result['moves']))
                self.assertTrue(any(m['type']=='pass' and m['button']==button for m in result['moves']))

    def test_live_state_keeps_opponent_take_compatible_with_search(self):
        import numpy as np
        from game_state.game import DurakGameState
        from game_state.bot import state_snapshot
        full=position([['6D','AC'],['8D','9D']],table=[{'attack':'6C'}])
        data={'button':'Pass','opponent':'I take','mine':'','deque':0,'hand':['6D','AC'],
              'field':{'cards':['6C'],'layout':[{'card':'6C','covers':None}]}}
        state=DurakGameState(out_cards=set(full['discard']),known_opponent_cards={'8D','9D'},
            detectors={key:lambda image,key=key:data[key] for key in data})
        state.update(np.zeros((2,2,3),np.uint8))
        engine=DurakEngine('S',iterations=100,time_limit_ms=0,seed=3)
        self.assertEqual(engine.suggest(state)['status'],'ok')
        self.assertEqual(engine.suggest(state_snapshot(state))['status'],'ok')

    def test_inconsistent_ocr_is_rejected(self):
        with self.assertRaises(ValueError):
            observation_from_state(dict(phase='defend_or_take',hand_cards=['7C'],
                field_cards=['6C'],field_layout=[{'card':None,'covers':None}]),trump='S')
        obs=observation(_native.Game.new_game(2,attacker=0).snapshot())
        obs['opponent_count']=7
        with self.assertRaises(ValueError): _native.analyze(obs)

    def test_search_releases_gil(self):
        obs=observation(_native.Game.new_game(2,attacker=0).snapshot())
        progress=[]
        started=threading.Event()
        stop=threading.Event()
        def counter():
            started.wait()
            while not stop.is_set():
                progress.append(1)
                stop.wait(.001)
        thread=threading.Thread(target=counter)
        thread.start()
        try:
            started.set()
            _native.analyze(obs,iterations=100000,time_limit_ms=80)
        finally:
            stop.set(); thread.join()
        self.assertGreater(len(progress),5)


if __name__=='__main__':
    unittest.main()
