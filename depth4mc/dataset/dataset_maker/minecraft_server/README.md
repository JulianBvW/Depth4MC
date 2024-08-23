# How to create the server

To create the dataset you need a set of screenshots from the game and their depth values. To get the depth information, a shader can be used as explainend in the `dataset_maker` README. To get the two types of images from the exact same coordinates every time, a server is used with the `mcpi` plugin, which can record the path you take while running around a world and then giving you a datapack (a kind of modification for the game) that let's you teleport to the recorded positions.

## Setting up the *spigot* server with plugins

[Spigot](https://getbukkit.org/download/spigot) is a modification of the normal Minecraft server which enables plugin (modification) support. You can download it to this folder using:

```bash
SERVER_DIR="depth4mc/dataset/dataset_maker/minecraft_server"
curl -o $SERVER_DIR/spigot-1.19.4.jar https://download.getbukkit.org/spigot/spigot-1.19.4.jar
```

[RaspberryJuice](https://www.spigotmc.org/resources/raspberryjuice.22724/update?update=302507) is the plugin that is used to communicate with python. Download it to the `plugins` folder from [here](https://www.spigotmc.org/resources/raspberryjuice.22724/download?version=312696).

## Start the server

You can start the server by changing to the `minecraft_server` directory and running this command: ```java -Xmx2G -jar spigot-*.jar nogui```.

**Note**: If you are using WSL and you run Python with it, start the server also on WSL by allowing the Minecraft Port in your WSL firewall and starting the server with an added argument: ```java -Xmx2G -Djava.net.preferIPv4Stack=true -jar spigot-*.jar nogui```.

The first time you start the server, it will prompt you to accepting the Minecraft EULA. Change the `false` to `true` in the generated `eula.txt` and run the server again.

You can now join the server via a 1.19.4 client on the IP `localhost`.

To stop the server just type `stop` into the console.

## Changing the world

To get more diverse images, build some houses on the world or use an existing world. To do that, rename a world (saved inside `.minecraft/saves`) to just `world` and move it inside the `minecraft_server` folder after deleting the old world folder if existant.