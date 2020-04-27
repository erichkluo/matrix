# matrix

Matrix is a lightweight modular framework for prototyping and backtesting trading strategies. It offers:
- Flexibiltiy to define strategy logics in seconds and compute the backtesting results instantly;
- Modular design with the algos so you can reuse the logics without reinventing the wheels;
- Nodes structure so that you can build strategy over strategies (over more strategies)!

Inspired from the bt project: https://github.com/pmorissette/bt

## Requirements

- Python3
- numpy
- pandas
- sklearn
- ffn

## Installation

Copy matrix folder to your python site packages folder or under your project.

## Examples

A simple Equity/Bond allocation strategy with trend overlay - demostration of how to construct strategy over strategies.
https://github.com/exted/matrix/blob/master/examples/StrategyExample.ipynb