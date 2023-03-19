"""
Script for dry-running the ITL environment with a student and a user-controlled
teacher (in Unity lingo, Behavior Type: Heuristics for teacher agent)
"""
import os
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from itl import ITLAgent
from itl.opts import parse_arguments


if __name__ == "__main__":
    opts = parse_arguments()
    agent = ITLAgent(opts)
    agent()
