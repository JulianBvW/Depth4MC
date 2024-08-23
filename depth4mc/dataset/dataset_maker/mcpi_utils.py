import math

def get_pose(mc):
    position_rel = mc.player.getPos()
    position_abs = position_rel.x-SPAWN_OFFSET[0], position_rel.y-SPAWN_OFFSET[1], position_rel.z-SPAWN_OFFSET[2]
    rotation = mc.player.getRotation()
    rotation *= -1 if list(mc.player.getDirection())[0] > 0 else 1
    pitch = mc.player.getPitch()

    pose = {
        'x': position_abs[0],
        'y': position_abs[1],
        'z': position_abs[2],
        'rot': rotation,
        'pit': pitch
    }

    return pose

SPAWN_OFFSET = (48, -76, 16)