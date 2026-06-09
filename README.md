# CHIP Modular System

Welcome! This README will explain the general structure of the CHIP Modular System, and how to use it appropriately for your goals.


## Requirements
- Make sure bash is installed on the system (consider [using WSL2 on Windows](https://learn.microsoft.com/en-us/windows/wsl/install)).
- The system uses [Docker Compose](https://docs.docker.com/compose/install/), make sure it is installed and functioning. Docker Desktop is recommended for good overview, using the WSL2 backend is also highly recommended.
  On Mac, remove the line containing ``"credsStore"`` from ``~/.docker/config.json``.
- A system that has a CPU with virtualization support.
- [OPTIONAL]: GPU with CUDA support, for running LLMs locally.


## Quick Start
For a quick start with default settings, just navigate to the root folder and use `./chip.sh start`. You may access the front end at `http://localhost:9000`.
We advize using a dedicated terminal rather than a built-in terminal from e.g. visual studio code to support all socket-connections.

## Architecture Overview
The system works with the notion of "core" modules, and "non-core" modules. There are five different types of core modules:
- Front End
- Triple Extractor
- Reasoner
- Response Generator
- Logger

Any module that doesn't classify as one of these module types, is considered not to be a core module. All core modules must expose a `/process` route, which they use to transfer JSON data to one another via POST requests. A description of the expected models for the bodies of the requests will come after this subsection.

The general communication structure is as follows:
```                         
                           Logger
                             /|\
      ________________________|_________________________
     |              |                 |                 |
Front End ==> Triple Extractor ==> Reasoner ==> Response Generator ==|
	/|\                                                              |
	 |-------------------------SSE-----------------------------------|
```

All core modules communicate with the logger for logging, but other than that the communication is a pre-determined chain. At the end, a Server-Sent-Event (SSE) is used to communicate back to the Front End that the system has finished processing and has a chat response ready.

Core modules may communicate with other non-core/side modules to query e.g. knowledge, or to cache things via Redis, but this is the core loop that can always assumed to be present.

## Models

These are the models that the core modules expect the JSON bodies to conform to, sent via a POST request to the `.../process` route of the next module in the chain.

### Triple Extractor
[POST] `/process`:
```JSON
{
    "patient_name": <string>,   // The name of the user currently chatting
    "sentence": <string>,       // The sentence that the user submitted
    "timestamp": <string>       // The time at which the user submitted the sentence (ISO format)
}
```

### Reasoner
[POST] `/process`:
```JSON
{
    "sentence_data": {
        "patient_name": <string>,   // The name of the user currently chatting
        "sentence": <string>,       // The sentence that the user submitted
        "timestamp": <string>       // The time at which the user submitted the sentence (ISO format)
    },
    "triples": [
        {
            "subject":<string>, 
            "object": <string>, 
            "predicate":<string>
        },
        ...
    ]
}
```

### Response Generator
[POST] `/process`:
```JSON
	{
        "sentence_data": {
            "patient_name": <string>,   // The name of the user currently chatting.
            "sentence": <string>,       // The sentence that the user submitted.
            "timestamp": <string>       // The time at which the user submitted the sentence (ISO format).
        },
		"type": <string: Q|A>,          // Whether the reasoner decided to give an answer (A) or to request more information (Q).
		"data": <dict>                  // A dict containing the output of the reasoner.
	}
```


### Front End
[POST] `/process`:
```JSON
    {
        "message": <string>             // The generated message.
    }
```

### Logger
The Logger module is special, in that its format is already pre-determined by Python's logging framework.

## General Usage and Configuration
The system has a set of pre-configured core modules that it will start up with, specified in `core-modules.yaml`. This file initially does not exist, but it will be created (from default values) along with other configuration files by running the `chip.sh` script without any subcommands, and it looks like this:

```YAML
logger_module: logger-default
frontend_module: front-end-quasar
response_generator_module: response-generator-gemini
reasoner_module: reasoning-demo
triple_extractor_module: text-to-triples-rule-based
```

The module names correspond to the name of the directory they reside in within `modules` directory.

A second configuration file that will be generated is `setup.env`. This environment file contains the url/port mappings for the modules, which are derived from their respective compose files. This is how the modules know how to reach each other. The mapping uses the following convention: `MODULE_NAME_CONVERTED_TO_CAPS_AND_UNDERSCORES=<module-name>:<module-port>`. 

Finally, every module also has its own `config.env` which will be copied from the default values in their `config.env.default` file. This `config.env` file is not tracked by git, so it is a good place to store things like API keys that modules may need. They will be available within the container as an environment variable.

Full command overview for the `chip.sh` script:


- `chip.sh` (*without args*): generates the `core-modules.yaml` file if it doesn't exist from `core-modules.yaml.default`, and then generates `setup.env` containing the module mappings. `setup.env` will always be overwritten if it already exists. This also takes place for any of the subcommands. It also generates `config.env` for any module that didn't have it yet, from their `config.env.default`. Configs are not git-tracked, only their defaults are.

- `chip.sh start [module name ...]`: builds and starts the system pre-configured with the current core modules specified in `core-modules.yaml` and their dependencies, or starts specific modules and their dependencies given by name and separated by spaces.

- `chip.sh build [module name ...]`: builds the core modules specified in `core-modules.yaml` and their dependencies, or builds specific modules and their dependencies given by name and separated by spaces.

- `chip.sh stop`: takes down any active modules.

- `chip.sh clean`: removes all volume data, giving you a clean slate.

- `chip.sh list`: prints a list of all available modules.

- `chip.sh auto-complete`: adds auto-completion for the modules to the script. If you prefix this command with `source`, it will immediately load the auto-complete definitions in the current terminal, otherwise you have to restart the terminal for it to take effect.


For instance, if you are in the process of creating a new module, and just want to build it, you would use `chip.sh build my-cool-new-module`. If you want to both build and start it, you would use `chip.sh start my-cool-new-module`. Say your module has `redis` and `knowledge-demo` as dependencies, then docker-compose will automatically also build and start the `redis` and `knowledge-demo` modules for you.

## Generic Module Structure
All modules adhere to the following structure:
```
my-cool-new-module
|- Dockerfile (optional)
|- compose.yml
|- README.md
|- ...
```

The [Dockerfile](https://docs.docker.com/reference/dockerfile/) can be omitted, if `compose.yml` specifies a particular image without further modifications. This may be the case for certain non-core modules, such as `redis`.

How it practically works, is that all `compose.yml` files of all modules will be collected by the `chip.sh` script, and then merged into a big docker compose configuration using docker compose [merge](https://docs.docker.com/compose/how-tos/multiple-compose-files/merge/).

Here's an example of a minimal `compose.yml` file of the `my-cool-new-module` module, residing in a directory of the same name within the `modules` directory:
```YAML
services:  # This is always present at the root.
  my-cool-new-module:  # SERVICE NAME MUST CORRESPOND TO DIRECTORY/MODULE NAME
    env_file: setup.env  # The module mappings, don't touch this.
    expose: 
    - 5000  # The backend port, generally no need to touch this if using Flask.
    build: ./modules/my-cool-new-module/.  # Build according to the dockerfile in the current folder, only fix the module name.
    volumes:
    - ./modules/my-cool-new-module/app:/app  # Bind mount, for syncing code changes, only fix the module name.
    depends_on: ["redis"]  # Modules that this module depends on and that will be started/built along with it.
```

Modules should generally use the Python Flask backend, which means that somewhere in the module's directory (often the root, but sometimes it is nested, e.g. see `front-end-quasar`) there will be an `app` directory, which is the Flask app. The Flask apps are always structured as follows:
```
app
|- tests...            --> The tests
|- util... 		      --> All custom utilities and code
|- routes.py 	      --> All the routes and communication related code
|- __init__.py  	      --> Flask initialization/setup
```

The idea is to keep a clean separation of concerns: functionality related code will only be found in `util`, while `routes.py` only concerns itself with route definitions and request sending.

The `requirements.txt` file should generally reside next to the `app` directory, which are both usually found in the root directory for most modules. Hence, the typical module looks like this:
```
my-cool-new-module
|- Dockerfile
|- compose.yml
|- README.md
|- app...
	|- tests...
	|- util...
	|- routes.py
	|- __init__.py
|- requirements.txt
```


## Extension Guide

The previous section should already have outlined most of the details regarding the module structure, but here is a quick guide to get you started right away.

1. The easiest way to get started is to just copy an existing module that most closely resembles what you want to create.

2. Then, think of a good name. The naming convention used for the core modules is as follows: `<MODULE_TYPE>-<NAME>`. If the module is non-core, then you can use any name, as long as it doesn't clash with the core module type names (e.g. the Redis module is just called `redis`). Say you want to make a new Reasoner, by the name "brain", then you call it `reasoner-brain`.

3. Rename everything that needs to be renamed:
    - [ ] The module directory name
	- [ ] The service name in the `compose.yml` file
	- [ ] Fix the paths for the volumes and the build directory in the `compose.yml` file.

4. Adjust/Add anything else you need in terms of configuration in the `compose.yml` file:
    - [ ] Did you expose all the ports you need?
	- [ ] Are there dependencies such as `redis`? Did you add them?
	- [ ] Do you need additional bind mount or data volumes?
    - [ ] Environment variables?
	
	Check https://docs.docker.com/compose/ for detailed configuration help.

5. Tweak the Dockerfile:
	- [ ] Change the base image, if you need something specific
	- [ ] Install specific packages that you may need
	- [ ] Install special requirements that you cannot put in `requirements.txt`, e.g. due to using a different package manager than pip for them.
    - [ ] Other custom/misc. system setup that you may need.

	Check https://docs.docker.com/reference/dockerfile/ for a detailed Dockerfile reference.

    You can build the module from the root folder using `./chip.sh build <MODULE_NAME>`.

6. Tweak the Python code to your liking:
	- [ ] Add your requirements to `requirements.txt`
	- [ ] Include your own code and utilities in `util`
	- [ ] Expose the routes you need in `routes.py` and call your code from `util`
	- [ ] Write your own tests
	- [ ] Change the module's appearance in the logs, by changing the service name in the `__init__.py` of the `ServiceNameFilter`

    If the bind mounts are setup properly in `compose.yml`, the code should automatically sync. If the module uses Flask, it will auto-reload the application whenever you make a change.

    You can run the module from the root folder using `./chip.sh start <MODULE_NAME>`.

7. Add any configuration you want to expose to `config.env.default`. They are made available as environment variables within the container, and the user may edit the generated `config.env` to tweak the settings you wish to make configurable. Think of things such as API keys, model name, performance settings, etc.

## Tests and CI
CI is setup for the project, and will run automatically for any module that has a `tests` folder within an `app` folder, which is generally the file structure that Flask adheres to. Modules that have no such folder will not be considered for the test runner.

## Citations
We kindly request you to cite this paper if you find it valuable for your research
```bibtex
@article{DENHENGST2026102536,
author = {den Hengst, Floris and Alaka, Shaad and Kamphorst, Bart A},
doi = {https://doi.org/10.1016/j.softx.2026.102536},
issn = {2352-7110},
journal = {SoftwareX},
keywords = { Artificial intelligence, Dialogue, Hybrid intelligence, Knowledge graphs,Lifestyle support},
pages = {102536},
title = {Collaborative hybrid intelligence platform {CHIP}: A modular architecture for developing and testing personalized lifestyle support interactions},
url = {https://www.sciencedirect.com/science/article/pii/S2352711026000300},
volume = {33},
year = {2026}
}
```
