# Creating the dataset

## Setup

### 1. Python Setup

In you Python 3.9 Conda environment, install the python package [mcpi](https://github.com/martinohanlon/mcpi).

### 2. Server Setup

See the README file in the `minecraft_server` folder for detailed instructions on how to setup the server.

### 3. Client Setup

Create a Minecraft 1.19.4 instance and install the modloader *[Fabric](https://fabricmc.net/)*. Copy the mods [Sodium](https://modrinth.com/mod/sodium/version/mc1.19.4-0.4.10) and [Iris Shaders](https://modrinth.com/mod/iris/version/1.6.11+1.19.4) to the games mods folder located at `.minecraft/mods`. One way to do this is using the third party launcher [MultiMC](https://multimc.org/) as it simplifies mod installation.

For the depth information, a *shaderpack* is used. Download it [here](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa0E3ZlVoTWVNMGRIaGhNUFBwbTNfTm1wZm9HZ3xBQ3Jtc0tsSThQcG1iQW5mNEhSeDFfWEJVOW5falgySDNIWDdIQmowMUdXemFaNEFYSUlUMDNyNnZackd1VVdCZVM3TFA1MV9fNVBaNUxDX2NmYVEzOXdvLWV6ZlpRaTVRTllxMU9BQ1lwcG1sblNIZHhOVFRFQQ&q=http%3A%2F%2Fwww.mediafire.com%2Ffile%2F5tan9hrgjhr3vu4%2FCPDepthMap.zip&v=nakyctgYDM8) and copy it into `.minecarft/shaderpacks`. In-game you can select it in the *Video Settings*. YouTube offers tutorials on how install and use *Iris Shaders*.

> [!NOTE]
> To get consistent image proportions, change the Minecraft window size to 854x480. On *MultiMC* this can be changed in *Settings -> Minecraft -> Window Size*.

### 4. mcpi Setup

mcpi has a wrong coordinate origin. To counter it, first double-tap the spacebar to go into fly mode. Then type `/tp 0.0 0.0 0.0` into the in-game chat accessible via *T* to teleport yourself to the world center. You are probably inside some blocks. Destroy them and run the in-game command again. Now run `python depth4mc/dataset/dataset_maker/mcpi_init.py` in a new console. In the Minecraft game chat it now show you your mcpi offset. Write those values in the `mcpi_utils.py` to the `SPAWN_OFFSET` variable. Run the init script again and check the values using the debug menu accessible via *F3*.

## Dataset creation

TODO