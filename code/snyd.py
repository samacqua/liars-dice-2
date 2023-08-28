"""
Contains the Game class, as well as various simple netural network architectures.
"""

import random
import torch
from torch import nn
import itertools
import numpy as np
import math
from collections import Counter

from typing import List, Tuple


class Net(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        hiddens = (100,) * 4

        # Bilinear can't be used inside nn.Sequantial
        # https://github.com/pytorch/pytorch/issues/37092
        self.layer0 = torch.nn.Bilinear(d_pri, d_pub, hiddens[0])

        layers = [torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        joined = self.layer0(priv, pub)
        return self.seq(joined)


class NetConcat(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        hiddens = (500, 400, 300, 200, 100)

        layers = [torch.nn.Linear(d_pri + d_pub, hiddens[0]), torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        if len(priv.shape) == 1:
            joined = torch.cat((priv, pub), dim=0)
        else:
            joined = torch.cat((priv, pub), dim=1)
        return self.seq(joined)


class NetCompBilin(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        hiddens = (100,) * 4

        middle = 500
        self.layer_pri = torch.nn.Linear(d_pri, middle)
        self.layer_pub = torch.nn.Linear(d_pub, middle)

        layers = [torch.nn.ReLU(), torch.nn.Linear(middle, hiddens[0]), torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        joined = self.layer_pri(priv) * self.layer_pub(pub)
        return self.seq(joined)


class Resid(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        y = self.conv(y)


class Net3(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        def conv(channels, size):
            return nn.Sequential(
                nn.Conv1d(1, channels, kernel_size=size),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=size),
                nn.BatchNorm1d(channels),
            )

        # Bilinear can't be used inside nn.Sequantial
        # https://github.com/pytorch/pytorch/issues/37092
        self.layer0 = torch.nn.Bilinear(d_pri, d_pub, hiddens[0])

        layers = [torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        joined = self.layer0(priv, pub)
        return self.seq(joined)


class Net2(torch.nn.Module):
    def __init__(self, d_pri, d_pub):
        super().__init__()

        channels = 20
        self.left = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=2),
            nn.ReLU(),
        )

        self.right = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, 1, kernel_size=2),
            nn.ReLU(),
        )

        layer_size = 100
        self.bilin = torch.nn.Bilinear(d_pri, d_pub, layer_size)

        layers = [torch.nn.ReLU()]
        for i in range(3):
            layers += [torch.nn.Linear(layer_size, layer_size), torch.nn.ReLU()]
        layers += [torch.nn.Linear(layer_size, 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        if len(priv.shape) == 1:
            assert len(pub.shape) == 1
            priv = priv.unsqueeze(0)
            pub = pub.unsqueeze(0)
        # x = self.left(priv.unsqueeze(-2)).squeeze(1) + priv
        x = priv
        y = self.right(pub.unsqueeze(-2)).squeeze(1) + pub
        mixed = self.bilin(x, y)
        return self.seq(mixed)


def calc_args(dice_per_player: List[int], sides: int, variant: str) -> Tuple[int, int, int, int, int, int, int]:
    """Generates the features of the game given the user-defined parameters."""

    # Public state = [History of player 1, History of player 2, ..., is player 1 turn?, is player 2 turn?, ...]
    # so the shape = 1 x (N_ACTIONS * n_players + n_players)
    # Private state = [Hand of player (1-hot), 1 hot encoding of who's private hand]
    # so the shape = 1 x (sides * n_dice + n_players)

    n_players = len(dice_per_player)

    # Maximum call is (d1 + d2 + ...) 6s.
    # So encode history of calls as 1-hot encodings for each player.
    N_ACTIONS = sum(dice_per_player) * sides
    if variant == "stairs":
        N_ACTIONS = 2 * sum(dice_per_player) * sides
    N_ACTIONS += 1  # + 1 more action for calling "Liar"

    LIE_ACTION = N_ACTIONS - 1     # The final index of the actions represents the call action.
    D_PUB_INFO_PER_PLAYER = N_ACTIONS

    # The size of the game history is the number of actions times the number of players.
    D_HIST = N_ACTIONS * n_players

    # Add extra features to keep track of who's to play next.
    # One-hot encoding of the player index.
    CUR_PLAYER_INDEX = D_HIST
    D_PUB_INFO = CUR_PLAYER_INDEX + n_players

    # One player technically may have a smaller private space than the others,
    # but we just take the maximum for simplicity.
    D_PRI_INFO = max(dice_per_player) * sides

    # And then features to describe from who's perspective we
    # are given the private information.
    PRI_INDEX = D_PRI_INFO
    D_PRI_INFO += n_players

    return D_PUB_INFO, D_PRI_INFO, N_ACTIONS, LIE_ACTION, CUR_PLAYER_INDEX, PRI_INDEX, D_PUB_INFO_PER_PLAYER


class Game:
    def __init__(self, model: torch.nn.Module, dice_per_player: List[int], sides: int, variant: str):

        self.model = model
        self.dice_per_player = dice_per_player
        self.SIDES = sides
        self.VARIANT = variant
        self.n_players = len(dice_per_player)

        (
            self.D_PUB,
            self.D_PRI,
            self.N_ACTIONS,
            self.LIE_ACTION,
            self.CUR_PLAYER_INDEX,
            self.PRI_INDEX,
            self.D_PUB_PER_PLAYER,
        ) = calc_args(dice_per_player, sides, variant)

    def get_cur_player_pub(self, pub_state: torch.Tensor) -> int:
        """Returns the index of the current player."""

        for i in range(self.n_players):
            if pub_state[self.CUR_PLAYER_INDEX + i] == 1:
                return i
        raise ValueError("No current player found in state.")
    
    def get_cur_player_priv(self, priv_state: torch.Tensor) -> int:
        """Returns the index of the current player."""

        for i in range(self.n_players):
            if priv_state[self.PRI_INDEX + i] == 1:
                return i
        raise ValueError("No current player found in state.")

    def print_pub_state(self, state: torch.Tensor) -> None:
        """Prints the state."""

        print("History of calls")
        game_over = False
        for p in range(self.n_players):
            print("Player", p)
            for i in range(self.N_ACTIONS):
                if state[self.D_PUB_PER_PLAYER * p + i] == 1:
                    if i == self.LIE_ACTION:
                        print("LIE")
                        game_over = True
                    else:
                        n, d = divmod(i, self.SIDES)
                        print(n+1, str(d+1) + "s" if n > 0 else "")

        if not game_over:
            print(f"\nCurrent player: {self.get_cur_player_pub(state)}")

    def make_regrets(self, priv: torch.Tensor, state: torch.Tensor, last_call: int) -> List[torch.Tensor]:
        """Calculates the regrets.

        priv: Private state, including the perspective for the scores.
        state: Public state.
        last_call: Last action taken by a player. Returned regrets will be for actions after this one.
        """

        if self.get_cur_player_pub(state) != self.get_cur_player_priv(priv):
            raise ValueError("Warning: Regrets are not with respect to current player.")

        # Number of child nodes.
        n_actions = self.N_ACTIONS - last_call - 1

        # One for the current state, and one for each child.
        # The last call has already been applied to the state, so only
        # apply the child actions.
        batch = state.repeat(n_actions + 1, 1)
        for i in range(n_actions):
            self._apply_action(batch[i + 1], i + last_call + 1)

        # The private state is the same for all children.
        priv_batch = priv.repeat(n_actions + 1, 1)

        # Calculate the estimated value of the current state (v)
        # and the estimated value of each child action (vs).
        v, *vs = list(self.model(priv_batch, batch))

        # Calculate the regrets (the max difference between the estimate of a child action and current action).
        return [max(vi - v, 0) for vi in vs]
 
    def evaluate_call(self, rolls: List[Tuple[int, ...]], last_call: int) -> bool:
        """Returns True if the there are more die than called, false otherwise."""

        # Calling lie immediately is an error, so we pretend the
        # last call was good to punish the player.
        if last_call == -1:
            return True
        
        # The calls are 1-hot-encoded, so parse the actual call.
        # For example, with six dice: 0 -> (1 1), 1 -> (1 2), 2 -> (1 3), ..., 6 -> (2 1s)...
        n, d = divmod(last_call, self.SIDES)
        n, d = n + 1, d + 1  # (0, 0) means 1 of 1s

        # Count the dice to see if the call was correct.
        cnt = Counter(rolls)
        if self.VARIANT == "normal":
            actual = cnt[d]
        if self.VARIANT == "joker":
            actual = cnt[d] + cnt[1] if d != 1 else cnt[d]
        if self.VARIANT == "stairs":
            assert len(rolls) == 2, "Stairs variant only implemented for 2 players."
            r1, r2 = rolls
            if all(r == i + 1 for r, i in zip(r1, range(self.SIDES))):
                actual += 2 * len(r1) - r1.count(d)
            if all(r == i + 1 for r, i in zip(r2, range(self.SIDES))):
                actual += 2 * len(r2) - r1.count(d)

        return actual >= n

    def policy(self, priv: torch.Tensor, state: torch.Tensor, last_call: int, eps: float = 0) -> List[float]:
        """Generates a policy with respect to the calculated regrets."""

        # Calculate the regrets and add noise.
        regrets = self.make_regrets(priv, state, last_call)
        for i in range(len(regrets)):
            regrets[i] += eps

        # If all regrets are 0, we return a uniform distribution.
        if sum(regrets) <= 0:
            return [1 / len(regrets)] * len(regrets)
        
        # Otherwise, we normalize the regrets.
        else:
            s = sum(regrets)
            return [r / s for r in regrets]

    def sample_action(self, priv: torch.Tensor, state: torch.Tensor, last_call: int, eps: float) -> int:
        """Samples an action from the policy."""

        # Sample an action from the policy.
        pi = self.policy(priv, state, last_call, eps)
        action = next(iter(torch.utils.data.WeightedRandomSampler(pi, num_samples=1)))

        return action + last_call + 1

    def apply_action(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """Applies the given action to the state."""
        new_state = state.clone()
        self._apply_action(new_state, action)
        return new_state

    def _apply_action(self, state: torch.Tensor, action: int) -> None:
        """Applies the given action to the state inplace."""
        
        cur = self.get_cur_player_pub(state)

        # Update the history of actions for the current player.
        state[action + cur * self.D_PUB_PER_PLAYER] = 1

        # Update the current player.
        state[self.CUR_PLAYER_INDEX + cur] = 0
        state[self.CUR_PLAYER_INDEX + (cur + 1) % self.n_players] = 1

        return state

    def make_priv(self, roll: Tuple[int, ...], player: int) -> torch.Tensor:
        """Returns the private state for the player."""

        priv = torch.zeros(self.D_PRI)
        priv[self.PRI_INDEX + player] = 1

        # New method inspired by Chinese poker paper to encode roll.
        # 1-hot encoding w/ number of dice. Example: if 2 dice, then
        # first 2 indices are for dice w/ 1 side. 
        cnt = Counter(roll)
        for face, c in cnt.items():
            for i in range(c):
                priv[(face - 1) * max(self.dice_per_player) + i] = 1

        return priv

    def make_init_state(self) -> torch.Tensor:
        """Returns the initial state of the game."""
        state = torch.zeros(self.D_PUB)
        state[self.CUR_PLAYER_INDEX] = 1
        return state

    def rolls(self, player: int) -> List[Tuple[int, ...]]:
        """Returns all possible rolls for the given player."""
        n_faces = self.dice_per_player[player]
        return [
            tuple(sorted(r))
            for r in itertools.product(range(1, self.SIDES + 1), repeat=n_faces)
        ]

    def get_calls(self, state: torch.Tensor) -> List[int]:
        """Get a list of calls made so far."""

        # Merge the calls of all players.
        # (Since the same call can't be made twice, summing across player histories
        # just gets a full history of the game).
        merged = sum([
            state[self.D_PUB_PER_PLAYER * i: self.D_PUB_PER_PLAYER * (i + 1)]  for i in range(self.n_players)
        ])
        return (merged == 1).nonzero(as_tuple=True)[0].tolist()

    def get_last_call(self, state: torch.Tensor) -> int:
        """Get the last call made."""
        ids = self.get_calls(state)
        if not ids:
            return -1
        return int(ids[-1])
