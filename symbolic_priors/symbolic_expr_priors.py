"""Symbolic expressions to be used as Inductive priors."""

import math


# Various constants used in symbolic expressions.
PI = 3.1417
SIGMA = 0.1112
LIGHT_SPEED = 186_000

ALL_SYMBOLIC_PRIMITIVE_CLASS = (
    "ad",  # Add.
    "mu",  # Multiply.
    "su",  # Substract.
    "dv",  # Divide.
    "sq",  # Squareroot.
    "py",  # Poly.
    "ex",  # Exponential.
    "ln",  # Log.
    "co",  # Cosine.
    "si",  # Sine.
    "ta",  # Tan.
    "as",  # ArcSin.
    "ac",  # ArcCos.
    "at",  # ArcTan.
)

ALL_SYMBOLIC_PRIMITIVE_DICT = {
    "ad":  'Add',
    "mu":  'Multiply',
    "su":  'Substract',
    "dv":  'Divide',
    "sq":  'Squareroot',
    "py":  'Polynomial',
    "ex":  'Exponential',
    "ln":  'Logorathmic',
    "co":  'Cosine',
    "si":  'Sine',
    "ta":  'Tangent',
    "as":  'ArcSine',
    "ac":  'ArcCosine',
    "at":  'ArcTangent',  
}

ALL_SYMBOLIC_PRIMITIVE_CLASS_DICT = {sym: int(i) for i, sym in enumerate(
    ALL_SYMBOLIC_PRIMITIVE_CLASS)}
ALL_CLASS_IND_SYMBOLIC_PRIMITIVE_DICT = {int(i):sym for i, sym in enumerate(
    ALL_SYMBOLIC_PRIMITIVE_CLASS)}

SYMBOLIC_PRIMITIVE_CLASS = (
    "sq",  # Squareroot.
    "py",  # Poly.
    "ln",  # Log.
    "co",  # Cosine.
    "si",  # Sine.
    "ta",  # Tan.
    "as",  # ArcSin.
    "ac",  # ArcCos.
    "at",  # ArcTan.
    "ab",  # Abs.
)


def gaussian_prob_density(x: float):
  """Computes guassian probability density for given input."""
  return (math.exp(-1 * (math.pow(x, 2)) / 2)) / 2 * PI


def nguyen1_poly_expression1(x: float):
  """Simple nguyen expression for polynomial - expression1."""
  return math.pow(x, 3) + math.pow(x, 2) + math.pow(x, 1)


def nguyen1_poly_expression2(x: float):
  """Simple nguyen expression for polynomial - expression2."""
  return math.pow(x, 4) + math.pow(x, 3) + math.pow(x, 2) + math.pow(x, 1)


def nguyen1_poly_trig_expression1(x: float):
  """Math expression1 with polynomial and trignometry functions."""
  return math.sin(math.pow(x, 2)) * math.cos(x) - 1


def nguyen1_poly_trig_expression2(x: float):
  """Math expression2 with polynomial and trignometry functions."""
  return math.sin(x) + math.sin(x + math.pow(x, 2))


def nguyen1_log_expression1(x: float):
  """Math expression1 with lograthemic functions."""
  return math.log(x+1, 10) + math.log(math.pow(x, 2) + 1)


def sthoppay_math_expression1(x: float):
  """Math expression with poly and trignometric primitives."""
  return math.cos(x) + math.sin(x + math.pow(x, 2))


def sthoppay_math_expression2(x: float):
  """Math expression with poly and trignometric primitives."""
  return (math.acos(math.pow(x, 2)) - math.tan(math.pow(x, 3)) + 
          math.pow(x, 2) + x)


def sthoppay_math_expression3(x: float):
  """Math expression with trignometric primitive."""
  return x * math.asin(x) / math.tan(x) * math.cos(x)


def sthoppay_math_expression4(x: float):
  """Math expression with logarithm and poly primitives."""
  return math.log(x + 1.4) + math.log(math.pow(x, 2) + 1.3)


def sthoppay_math_expression5(x: float):
  """Math expression with logarithm and trig primitives."""
  return math.log(x) / math.atan(x)


def sthoppay_math_expression6(x: float):
  """Math expressino with logarithm and algebric primitives."""
  return math.log(x) * math.sqrt(math.pow(x, 2) + 1)


def sthoppay_math_expression7(x: float):
  """Math expression with trignometric and algebric primitives."""
  return 6 * math.sin(x) * math.cos(x)


def sthoppay_math_expression8(x: float):
  """Math expression with simple arithmetic primitive."""
  return math.sqrt(x)


def sthoppay_math_expression9(x: float):
  """Math expression with poly and exponential primitives."""
  return math.pow(x, 3) / math.exp(math.pow(x, 2))


def sthoppay_math_expression10(x: float):
  """Math expression with exponential and trignometric primitives."""
  return math.exp(math.pow(x, 3)) + math.sin(x)


def gaussian_prob_density_2d(theta: float, theta1: float):
  """Computes guassian probability density for given relative input."""
  # pylint: disable=line-too-long
  return (math.exp(-1 * (math.pow(theta - theta1, 2)) / (2 * math.pow(SIGMA, 2)))) / math.sqrt(2 * PI * math.pow(SIGMA, 2))
  # pylint: enable=line-too-long


def relative_mass(mass: float, velocity: float):
  """Computes mass of a object in relativity constraints."""
  # pylint: disable=line-too-long
  return mass / (math.sqrt(1 - math.pow(velocity, 2)/math.pow(LIGHT_SPEED, 2)))
  # pylint: enable=line-too-long
