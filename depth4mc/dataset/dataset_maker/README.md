# Creating the dataset

## Setup

### 1. Python Setup

In you Python 3.9 Conda environment, install the python package [mcpi](https://github.com/martinohanlon/mcpi).

### 2. Server Setup

See the README file in the `minecraft_server` folder for detailed instructions on how to setup the server.

### 3. Client Setup

Create a Minecraft 1.19.4 instance and install the modloader *[Fabric](https://fabricmc.net/)*. Copy the mods [Sodium](https://modrinth.com/mod/sodium/version/mc1.19.4-0.4.10) and [Iris Shaders](https://modrinth.com/mod/iris/version/1.6.11+1.19.4) to the games mods folder located at `.minecraft/mods`. One way to do this is using the third party launcher [MultiMC](https://multimc.org/) as it simplifies mod installation.

For the depth information, a *shaderpack* is used. Download it [here](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa0E3ZlVoTWVNMGRIaGhNUFBwbTNfTm1wZm9HZ3xBQ3Jtc0tsSThQcG1iQW5mNEhSeDFfWEJVOW5falgySDNIWDdIQmowMUdXemFaNEFYSUlUMDNyNnZackd1VVdCZVM3TFA1MV9fNVBaNUxDX2NmYVEzOXdvLWV6ZlpRaTVRTllxMU9BQ1lwcG1sblNIZHhOVFRFQQ&q=http%3A%2F%2Fwww.mediafire.com%2Ffile%2F5tan9hrgjhr3vu4%2FCPDepthMap.zip&v=nakyctgYDM8) and copy it into `.minecarft/shaderpacks`. In-game you can select it in the *Video Settings*. YouTube offers tutorials on how install and use *Iris Shaders*.

The depth shader has a resolution of 256 integer values (0 eaning nearest, 255 meaning furthest away). Thus, if your set the clipping (the point from where further distances are just encoded as 255) to a high value, to get the details in the distant blocks, you loose details right in front of you. So, extract the archive of the depth shader and create a copy of the shader folder. Rename one to `CPDepthMapNear` and the other to `CPDepthMapFar`. Delete the archive. In both shader folders, modify the file `shaders/final.fsh` by setting line four to `#define clipping 6` for the near depth shader and `#define clipping 60` for the far one.

> [!NOTE]
> To get consistent image proportions, change the Minecraft window size to 854x480. On *MultiMC* this can be changed in *Settings -> Minecraft -> Window Size*.

### 4. mcpi Setup

mcpi has a wrong coordinate origin. To counter it, first double-tap the spacebar to go into fly mode. Then type `/tp 0.0 0.0 0.0` into the in-game chat accessible via *t* to teleport yourself to the world center. You are probably inside some blocks. Destroy them and run the in-game command again. Now run `python depth4mc/dataset/dataset_maker/mcpi_init.py` in a new console. In the Minecraft game chat it now show you your mcpi offset. Write those values in the `mcpi_utils.py` to the `SPAWN_OFFSET` variable. Run the init script again and check the values using the debug menu accessible via *F3*.

## Dataset creation

The process consists of three steps: recording the coordinates you want to take pictures of, taking the screenshots, and finally taking the screenshots again but with a depth filter to get ground truth depth information.

### 1. Recording coordinates

Run `python depth4mc/dataset/dataset_maker/mcpi_record_poses.py <NUM_IMGS>` in a new console. After five seconds, the script starts recording your in-game pose every 10th of a second resulting in 3000 recorded poses (or whatever specified number). Fly or run around your world visiting normal terrain and buildings to get a diverse set of images. After the duration of the script, it will generate a *datapack* (a modification for the game that enables us to visit the exact coordinates as recorded) and copies it to the server. An output folder will also be generated whith the time of the run.

> [!WARNING]
> As the datapack uses right-click detection for teleportation, don't look at anything that could be interacted with via a right click like chests, doors, flower pots, levers, ...

### 2. Taking screenshots

To load the (new) datapack, type `/reload` into the in-game chat. It should now display the time of the correct run in the in-game chat.
Take a *Carrot on a Stick* from the creative inventory and hold it in your hand. If you were to use (to 'right-click') it, the script would teleport you to the next recorded position. Go into fly mode and press *F1* to disable the GUI.

The idea is now to teleport, then take a screenshot and then teleport again and so on. To automate this process, go into the key binding settings (ESC -> Options -> Controls -> Key Binds) and change the key for *Use Item/Place Block* to *y* and *Take Screenshot* to *u*. Now you can use for example a PowerShell script to automate the process. An example of this script can be found at the bottom of this file.

Run the PowerShell (or alternative) script and focus the Minecraft window. This will now generate all screenshots to the screenshot folder. Using MultiMC you can access this folder by clicking *Minecraft Folder* in the launcher and then going into the *screenshots* folder. Move (not copy) all the screenshots from this folder to the *screenshots* folder in the current run folder generated by the `mcpi_record_poses.py` script.

### 3. Taking depth screenshots

To generate the depth info, we do the same thing as in the prior step, but with the two depth shaders. Click *o* to access the shader selector. Select the depth shader `far` and click *Done*. Run the PowerShell script again and copy the generated screenshots into the `depth_labels_far` folder from the `mcpi_record_poses.py` script.

After that rerun this step whit the other depth shader and copy those results to the `depth_labels_far` folder.

### 4. Finishing up

Do the past 3 steps as much as you like on different areas of the world or even different worlds. Then, run `python depth4mc/dataset/dataset_maker/convert_to_dataset.py` to combine all runs into one dataset that will be saved at `depth4mc/dataset/data`.

> [!WARNING]
> This will delete the old dataset from `depth4mc/dataset/data`!

## PowerShell Automation Script

Replace the `3000` with the number of screenshots you want to take!

```powershell
Start-Sleep -Seconds 5
for ($i = 0; $i -lt 3000; $i++) {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.SendKeys]::SendWait("y")
    Start-Sleep -Milliseconds 500
    [System.Windows.Forms.SendKeys]::SendWait("u")
    Start-Sleep -Milliseconds 500
    echo "$i"
}
```