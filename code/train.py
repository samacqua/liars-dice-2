"""
Train a new model from self-play.
"""

import random
import torch
from torch import nn
import itertools
import math
from collections import Counter
import argparse
import os

from snyd import *

parser = argparse.ArgumentParser()
parser.add_argument("d", type=int, nargs="+", help="Number of dice for players")
parser.add_argument("--sides", type=int, default=6, help="Number of sides on the dice")
parser.add_argument(
    "--variant", type=str, default="normal", help="one of normal, joker, stairs"
)
parser.add_argument(
    "--eps", type=float, default=1e-2, help="Added to regrets for exploration"
)
parser.add_argument(
    "--layers", type=int, default=4, help="Number of fully connected layers"
)
parser.add_argument(
    "--layer-size", type=int, default=100, help="Number of neurons per layer"
)
parser.add_argument("--lr", type=float, default=1e-3, help="LR = lr/t")
parser.add_argument("--w", type=float, default=1e-2, help="weight decay")
parser.add_argument(
    "--path", type=str, default="model.pt", help="Where to save checkpoints"
)

parser.add_argument("--N", type=int, default=10_000_000, help="Number of games to simulate / train for.")

args = parser.parse_args()


# Check if there is a model we should continue training
if os.path.isfile(args.path):
    checkpoint = torch.load(args.path)
    print(f"Using args from {args.path}")
    old_path = args.path
    args = checkpoint["args"]
    args.path = old_path
else:
    checkpoint = None

# Model : (private state, public state) -> value
D_PUB, D_PRI, *_ = calc_args(args.d, args.sides, args.variant)
model = NetConcat(D_PRI, D_PUB)
# model = Net(D_PRI, D_PUB)
# model = Net2(D_PRI, D_PUB)
game = Game(model, args.d, args.sides, args.variant)

if checkpoint is not None:
    print("Loading previous model for continued training")
    model.load_state_dict(checkpoint["model_state_dict"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@torch.no_grad()
def play(rolls: Tuple[Tuple, ...], replay_buffer: List[Tuple[torch.Tensor, torch.Tensor, float]]):
    """Plays a game and saves the results in replay_buffer.
    
    A game's result is 1 for all players who do not lose a die and -1 for the player who does.
    """

    priv_states = [game.make_priv(r, i).to(device) for i, r in enumerate(rolls)]

    def play_inner(pub_state):

        # Get the current player and the history of calls.
        cur = game.get_cur_player_pub(pub_state)    
        calls = game.get_calls(pub_state)
        assert cur == len(calls) % len(rolls), "The current player doesn't align with the game history."

        # Round ends when someone calls a lie.
        if calls and calls[-1] == game.LIE_ACTION:

            # The loser is the player who called the lie if there are enough die otherwise it is the person who lied.
            prev_call = calls[-2] if len(calls) >= 2 else -1
            if game.evaluate_call(rolls, prev_call):
                loser = cur
            else:
                loser = (cur - 1) % len(rolls)

        else:
            last_call = calls[-1] if calls else -1
            action = game.sample_action(priv_states[cur], pub_state, last_call, args.eps)
            new_state = game.apply_action(pub_state, action)
            
            # Just classic min/max stuff.
            loser = play_inner(new_state)

        # Save the result from the perspective of all sides.
        for player in range(len(rolls)):
            res = 1 if player != loser else -1
            replay_buffer.append((priv_states[player], pub_state, res))

        return loser

    with torch.no_grad():
        pub_state = game.make_init_state().to(device)
        play_inner(pub_state)


def print_strategy(state: torch.Tensor) -> None:
    """Prints the strategy for the player who's turn it is."""

    player = game.get_cur_player_pub(state)
    last_call = game.get_last_call(state)

    total_v = 0
    total_cnt = 0
    for roll, cnt in sorted(Counter(game.rolls(player)).items()):

        # For the possible roll, calculate the value and normalized regret of each possible action.
        priv = game.make_priv(roll, player).to(device)
        v = model(priv, state)
        rs = torch.tensor(game.make_regrets(priv, state, last_call=last_call))
        if rs.sum() != 0:
            rs /= rs.sum()

        # Make a string of the strategy, where the strategy is choosing an action proportional
        # to the regret.
        strat = []
        for action, prob in enumerate(rs):
            n, d = divmod(action, game.SIDES)
            n, d = n + 1, d + 1
            if d == 1:
                strat.append(f"{n}:")   # The number of dice.
            strat.append(f"{prob:.2f}") # The probability of calling side n.

        # Print the strategy for the roll: "roll: value of roll (count of roll because of duplicates)"...
        # "number of dice: probability of calling side 1, probability of calling side 2, ..."
        print(roll, f"{float(v):.4f}".rjust(7), f"({cnt})", " ".join(strat))
        total_v += v
        total_cnt += cnt

    print(f"Mean value: {(total_v / total_cnt).item():.4f}")


class ReciLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma=1, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(ReciLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            base_lr / (self.last_epoch + 1) ** self.gamma
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr / (self.last_epoch + 1) ** self.gamma for base_lr in self.base_lrs
        ]


def train():

    torch.manual_seed(0)
    random.seed(0)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.w)
    scheduler = ReciLR(optimizer, gamma=0.5)
    value_loss = torch.nn.MSELoss()
    all_rolls = list(itertools.product(*[game.rolls(i) for i, n_dice in enumerate(args.d)]))

    n_games_per_iter = 100
    n_iters = args.N // n_games_per_iter

    for t in range(n_iters):
        replay_buffer = []

        # Play games from a random sample of the possible starting rolls.
        for rolls in (
            all_rolls if len(all_rolls) <= n_games_per_iter else random.sample(all_rolls, n_games_per_iter)
        ):
            play(rolls, replay_buffer)

        # Train on the replay buffer.
        random.shuffle(replay_buffer)
        privs, states, y = zip(*replay_buffer)

        privs = torch.vstack(privs).to(device)
        states = torch.vstack(states).to(device)
        y = torch.tensor(y, dtype=torch.float).reshape(-1, 1).to(device)

        y_pred = model(privs, states)

        # Compute and print loss.
        loss = value_loss(y_pred, y)
        print(t, loss.item())

        # Print the starting move strategy every 5 iterations.
        if t % 5 == 0:
            with torch.no_grad():
                print_strategy(game.make_init_state().to(device))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Save the model every 10 iterations.
        if args.path and (t + 1) % 10 == 0:
            print(f"Saving to {args.path}")
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": args,
                },
                args.path,
            )

        # Save the model checkpoint every 1000 iterations.
        if args.path and (t + 1) % 1000 == 0:
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": args,
                },
                f"{args.path}.cp{t+1}",
            )


train()
