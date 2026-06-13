from elastic.sync import config

SIMPLE_TYPES = ["kick", "control", "out"]
SHOT_TYPES = ["shot", "shot_block"]
PASS_LIKE_OPEN = config.PASS_LIKE_OPEN
SET_PIECE = config.SET_PIECE
OUTGOING_TYPES = PASS_LIKE_OPEN + SET_PIECE + ["tackle", "bad_touch", "dispossessed", "kick"]
INCOMING_TYPES = config.INCOMING