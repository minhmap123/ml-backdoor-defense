from .badnets import BadNets
from .catback import CatBackAttacker
from .none import NoneAttack
from .tabdoor import TabDoor
from .base import AttackResult

ATTACK_REGISTRY = {
    "badnets": BadNets,
    "catback": CatBackAttacker,
    "none": NoneAttack,
    "tabdoor": TabDoor,
}


def get_attack(cfg, attack_name=None):
    name = (attack_name or cfg.name).lower()
    try:
        return ATTACK_REGISTRY[name](cfg)
    except KeyError as exc:
        available = ", ".join(sorted(ATTACK_REGISTRY))
        raise ValueError(f"Unknown attack: {name}. Available attacks: {available}") from exc
