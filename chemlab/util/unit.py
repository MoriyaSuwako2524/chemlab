import numpy as np
# -----------------------------
# Unit system (scalar quantities)
# -----------------------------
class unit_type:
    """Base class for scalar unit types.

    Parameters
    ----------
    value : array-like
        Numeric value(s); stored as NumPy array of dtype float.
    unit : str
        The current unit key (must exist in self.DICT).

    Attributes
    ----------
    DICT : dict
        Maps unit name -> scale relative to the canonical unit for the category.
    modify_value : bool
        If True, `convert_to` updates in-place; otherwise returns a converted copy.
    """

    category = None

    def __init__(self, value, unit):
        self.value = np.array(value, dtype=float)
        self.unit = unit
        self.DICT = {}
        self.modify_value = False

    def convert_to(self, target):
        """Convert to another unit within the same category.

        Parameters
        ----------
        target : str
            Target unit name.

        Returns
        -------
        np.ndarray
            Converted values (and updates internal state if `modify_value` is True).
        """
        factor = self.DICT[target] / self.DICT[self.unit]
        value = self.value * factor
        if self.modify_value:
            self.value = value
            self.unit = target
        return value


class ENERGY(unit_type):
    """Energy units. Canonical unit: Hartree."""

    category = "energy"

    def __init__(self, value, unit="hartree"):
        super().__init__(value, unit)
        self.DICT = {"Hartree":1,"hartree": 1, "kcal/mol": 627.51, "kcal": 627.51, "ev": 27.2113863, "kj": 2625.5}


class DISTANCE(unit_type):
    """Distance units. Canonical unit: Ångström."""

    category = "distance"

    def __init__(self, value, unit="ang"):
        super().__init__(value, unit)
        self.DICT = {"Ang":1, "ang": 1, "bohr": 1 / 0.529177, "nm": 10}


class MASS(unit_type):
    """Mass units. Canonical unit: atomic mass unit (amu)."""

    category = "mass"

    def __init__(self, value, unit="amu"):
        super().__init__(value, unit)
        self.DICT = {"amu": 1, "g": 1.66054e-24, "kg": 1.66054e-27}


class TIME(unit_type):
    """Time units. Canonical unit: femtosecond (fs)."""

    category = "time"

    def __init__(self, value, unit="fs"):
        super().__init__(value, unit)
        self.DICT = {"fs": 1, "ps": 1e-3, "ns": 1e-6, "s": 1e-15}


# --------------------------------
# Unit system (composite quantities)
# --------------------------------
class complex_unit_type:
    """Base class for composite unit types consisting of multiple categories.

    Parameters
    ----------
    value : array-like
        Numeric value(s) stored as float array.
    units : dict
        Mapping from category -> (unit_name, power) describing the dimensionality.

    Example
    -------
    For a force/gradient: {"energy": ("hartree", 1), "distance": ("bohr", -1)}
    """

    def __init__(self, value, units: dict):
        self.value = np.array(value, dtype=float)
        self.units = units
        self.modify_value = False

    def convert_to(self, target_units: dict):
        """Convert a composite quantity to another set of units.

        `target_units` must use the same categories and exponents.
        """
        factor = 1.0
        for category, (unit, power) in self.units.items():
            base = UNIT_REGISTRY[category](1.0, unit)
            target_unit = target_units[category][0]
            factor *= (base.convert_to(target_unit)) ** power
        value = self.value * factor
        if self.modify_value:
            self.value = value
            self.units = target_units
        return value

    def generate_all_conversions(self):
        """Generate conversion results for all unit combinations of involved categories.

        Returns
        -------
        dict
            Maps a compact key like "kcal/mol * ang^-1" to converted values.
        """
        unit_lists = []
        categories = []
        for category, (unit, power) in self.units.items():
            categories.append(category)
            unit_lists.append(list(UNIT_REGISTRY[category](1, unit).DICT.keys()))

        results = {}
        for combo in product(*unit_lists):
            target_units = {cat: (u, self.units[cat][1]) for cat, u in zip(categories, combo)}
            key = " * ".join([
                f"{u}^{p}" if p != 1 else u for (_, (u, p)) in target_units.items()
            ])
            results[key] = self.convert_to(target_units)
        return results


class CHARGE(unit_type):
    """Charge units. Canonical unit: elementary charge (e)."""

    category = "charge"

    def __init__(self, value, unit="e"):
        super().__init__(value, unit)
        self.DICT = {
            "e": 1.0,                      # 1 e = 1 e
            "C": 1.0 / 1.602176634e-19,   # 1 C = 6.241509074e18 e
        }


class DIPOLE(complex_unit_type):
    """Dipole moment with common conversions (Debye, e·bohr, e·Å, C·m)."""

    category = "dipole"

    def __init__(self, value, charge_unit="e", distance_unit="bohr"):
        super().__init__(np.array(value, dtype=float), {
            "charge": (charge_unit, 1),
            "distance": (distance_unit, 1)
        })
        # Conversion anchors: 1 Debye = 0.393430307 e·Å = 0.20819434 e·bohr
        self.DICT = {
            "Debye": 1.0,            # treat Debye as canonical for convenience
            "e*bohr": 1.0 / 0.20819434,
            "au" : 1.0 / 0.20819434,
            "e*ang":  1.0 / 0.393430307,
            "C*m":    1.0 / 3.33564e-30,
        }


class FORCE(complex_unit_type):
    """
    WARNING!!!!!!! THIS CONVERT GRADIENT TO FORCE BY DEFAULT
    YOU NEED TO INPUT GRADIENT
    Force = -∂E/∂R. Stored with a leading minus sign relative to gradients.

    Note
    ----
    The input `value` is negated internally to adhere to the physics sign convention for forces.
    """

    def __init__(self, value, energy_unit="hartree", distance_unit="bohr"):
        super().__init__( -np.array(value, dtype=float), {  # Note the minus sign
            "energy": (energy_unit, 1),
            "distance": (distance_unit, -1)
        })


class GRADIENT(complex_unit_type):
    """Gradient = ∂E/∂R (positive sign)."""

    def __init__(self, value, energy_unit="hartree", distance_unit="bohr"):
        super().__init__( np.array(value, dtype=float), {
            "energy": (energy_unit, 1),
            "distance": (distance_unit, -1)
        })


# Registry for converting by category name
UNIT_REGISTRY = {
    "energy": ENERGY,
    "distance": DISTANCE,
    "mass": MASS,
    "time": TIME,
    "charge": CHARGE,
    "dipole": DIPOLE,
    "force": FORCE,
    "gradient": GRADIENT,
}
