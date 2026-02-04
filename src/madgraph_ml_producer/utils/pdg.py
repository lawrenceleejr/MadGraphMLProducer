"""
PDG particle ID utilities for MadGraphMLProducer.

Provides mappings and utilities for working with Particle Data Group IDs.
"""

from typing import Optional


class PDGInfo:
    """Particle Data Group ID information and utilities"""

    # Standard Model particles
    PARTICLES = {
        # Quarks
        1: ("d", "down quark"),
        2: ("u", "up quark"),
        3: ("s", "strange quark"),
        4: ("c", "charm quark"),
        5: ("b", "bottom quark"),
        6: ("t", "top quark"),
        # Leptons
        11: ("e-", "electron"),
        12: ("nu_e", "electron neutrino"),
        13: ("mu-", "muon"),
        14: ("nu_mu", "muon neutrino"),
        15: ("tau-", "tau"),
        16: ("nu_tau", "tau neutrino"),
        # Gauge bosons
        21: ("g", "gluon"),
        22: ("gamma", "photon"),
        23: ("Z", "Z boson"),
        24: ("W+", "W boson"),
        25: ("h", "Higgs boson"),
        # SUSY particles (SLHA2 conventions)
        1000001: ("~d_L", "down squark L"),
        1000002: ("~u_L", "up squark L"),
        1000003: ("~s_L", "strange squark L"),
        1000004: ("~c_L", "charm squark L"),
        1000005: ("~b_1", "sbottom 1"),
        1000006: ("~t_1", "stop 1"),
        1000011: ("~e_L", "selectron L"),
        1000012: ("~nu_eL", "electron sneutrino"),
        1000013: ("~mu_L", "smuon L"),
        1000014: ("~nu_muL", "muon sneutrino"),
        1000015: ("~tau_1", "stau 1"),
        1000016: ("~nu_tauL", "tau sneutrino"),
        1000021: ("~g", "gluino"),
        1000022: ("~chi_10", "neutralino 1"),
        1000023: ("~chi_20", "neutralino 2"),
        1000024: ("~chi_1+", "chargino 1"),
        1000025: ("~chi_30", "neutralino 3"),
        1000035: ("~chi_40", "neutralino 4"),
        1000037: ("~chi_2+", "chargino 2"),
        2000001: ("~d_R", "down squark R"),
        2000002: ("~u_R", "up squark R"),
        2000003: ("~s_R", "strange squark R"),
        2000004: ("~c_R", "charm squark R"),
        2000005: ("~b_2", "sbottom 2"),
        2000006: ("~t_2", "stop 2"),
        2000011: ("~e_R", "selectron R"),
        2000013: ("~mu_R", "smuon R"),
        2000015: ("~tau_2", "stau 2"),
    }

    # MadGraph particle names (for process strings)
    MG_NAMES = {
        1000021: "go",  # gluino
        1000022: "n1",  # neutralino 1
        1000023: "n2",
        1000024: "x1+",  # chargino 1
        1000025: "n3",
        1000035: "n4",
        1000037: "x2+",
        1000001: "dl",  # down squark L
        1000002: "ul",
        2000001: "dr",  # down squark R
        2000002: "ur",
        1000003: "sl",
        1000004: "cl",
        2000003: "sr",
        2000004: "cr",
        1000005: "b1",
        1000006: "t1",
        2000005: "b2",
        2000006: "t2",
    }

    # Invisible particles (for MET calculation)
    INVISIBLE_PDGS = {12, -12, 14, -14, 16, -16, 1000022, 1000023, 1000025, 1000035}

    # Quark PDG IDs
    QUARKS = {1, 2, 3, 4, 5, 6}
    LIGHT_QUARKS = {1, 2, 3, 4, 5}  # Excluding top

    # Lepton PDG IDs
    CHARGED_LEPTONS = {11, 13, 15}
    NEUTRINOS = {12, 14, 16}

    @classmethod
    def get_name(cls, pdg_id: int) -> Optional[str]:
        """Get particle name from PDG ID"""
        abs_id = abs(pdg_id)
        if abs_id in cls.PARTICLES:
            name = cls.PARTICLES[abs_id][0]
            if pdg_id < 0 and abs_id not in {21, 22, 23, 25}:  # Not self-conjugate
                # Handle antiparticles
                if name.endswith("+"):
                    return name.replace("+", "-")
                elif name.endswith("-"):
                    return name.replace("-", "+")
                else:
                    return name + "~" if "~" not in name else name.replace("~", "~*")
            return name
        return None

    @classmethod
    def get_description(cls, pdg_id: int) -> Optional[str]:
        """Get particle description from PDG ID"""
        abs_id = abs(pdg_id)
        if abs_id in cls.PARTICLES:
            desc = cls.PARTICLES[abs_id][1]
            if pdg_id < 0 and abs_id not in {21, 22, 23, 25}:
                return "anti-" + desc
            return desc
        return None

    @classmethod
    def is_invisible(cls, pdg_id: int) -> bool:
        """Check if particle is invisible (contributes to MET)"""
        return abs(pdg_id) in cls.INVISIBLE_PDGS

    @classmethod
    def is_quark(cls, pdg_id: int) -> bool:
        """Check if particle is a quark"""
        return abs(pdg_id) in cls.QUARKS

    @classmethod
    def is_light_quark(cls, pdg_id: int) -> bool:
        """Check if particle is a light quark (not top)"""
        return abs(pdg_id) in cls.LIGHT_QUARKS

    @classmethod
    def is_charged_lepton(cls, pdg_id: int) -> bool:
        """Check if particle is a charged lepton"""
        return abs(pdg_id) in cls.CHARGED_LEPTONS

    @classmethod
    def is_neutrino(cls, pdg_id: int) -> bool:
        """Check if particle is a neutrino"""
        return abs(pdg_id) in cls.NEUTRINOS

    @classmethod
    def is_gluon(cls, pdg_id: int) -> bool:
        """Check if particle is a gluon"""
        return pdg_id == 21

    @classmethod
    def get_mg_name(cls, pdg_id: int) -> Optional[str]:
        """Get MadGraph particle name from PDG ID"""
        abs_id = abs(pdg_id)
        if abs_id in cls.MG_NAMES:
            name = cls.MG_NAMES[abs_id]
            if pdg_id < 0:
                return name + "~" if not name.endswith("+") else name.replace("+", "-")
            return name
        return None
