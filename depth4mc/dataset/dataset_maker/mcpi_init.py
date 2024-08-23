import mcpi.minecraft as minecraft
from depth4mc.dataset.dataset_maker.mcpi_utils import get_pose

mc = minecraft.Minecraft.create()

mc.postToChat('#----------------------#')
mc.postToChat('')

mc.postToChat('Your mcpi (relative) position:')
p = mc.player.getPos()
mc.postToChat(str(round(p.x, 2)) + ' ' + str(round(p.y, 2)) + ' ' + str(round(p.z, 2)))
mc.postToChat('')

mc.postToChat('After setting up mcpi offset in mcpi_utils.py, this should now be your absolute coordinates from the F3 menu:')
p = get_pose(mc)
mc.postToChat(str(round(p['x'], 2)) + ' ' + str(round(p['y'], 2)) + ' ' + str(round(p['z'], 2)))
mc.postToChat('rot: ' + str(round(p['rot'], 2)) + ' pit: ' + str(round(p['pit'], 2)))
mc.postToChat('')
