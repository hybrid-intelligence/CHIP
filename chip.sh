#!/bin/bash
if [ ! -f ./core-modules.yaml ]; then
    echo "No core module configuration found, creating one from defaults..."
    cp ./core-modules.yaml.default ./core-modules.yaml
fi

for dir in ./modules/*; do
    str+=" -f ${dir}/compose.yml"
    if [ ! -f $dir/config.env ]; then
        echo "No module configuration found for module ${dir%%:*}, creating one from defaults..."
        cp $dir/config.env.default $dir/config.env
    fi
done

docker compose -f docker-compose-base.yml ${str} config > /tmp/chiptemp
modules=($(docker run --rm -v "/tmp":/workdir mikefarah/yq  '.services.* | key + ":" + .expose[0]' chiptemp))
rm /tmp/chiptemp

core_modules=($(docker run --rm -v "${PWD}":/workdir mikefarah/yq '.* | key + ":" + .' core-modules.yaml))

# If setup.env does not exist, create an empty setup.env
[ -f setup.env ] || touch setup.env
> setup.env
for module in ${modules[@]}; do
    name=${module%%:*}
    name=${name//-/_}
    name=$( echo $name | tr '[:lower:]' '[:upper:]')
    echo ${name}=${module} >> setup.env
    for core_module in ${core_modules[@]}; do
        core=${core_module%%:*}
        core=$( echo $core | tr '[:lower:]' '[:upper:]')
        core_name=${core_module##*:}
        core_name=${core_name//-/_}
        core_name=$( echo $core_name | tr '[:lower:]' '[:upper:]')
        if [[ "$core_name" == "$name" ]]; then
            echo $core=$name >> setup.env
        fi
    done
done
echo "Created module-address and core-module mappings in setup.env..."

modules_up=""
for core_module in ${core_modules[@]}; do
        modules_up+=${core_module##*:}" "
done


if [[ -z "$1" ]] ; then
echo "Finished setting up!"
exit 0
fi

case $1 in
  build)
    if [[ -z "$2" ]] ; then
        echo "Building core modules:"
        echo $modules_up
        docker compose -f docker-compose-base.yml ${str} build ${modules_up}
    else
        echo "Building specific modules: "+"${@:2}"
        docker compose -f docker-compose-base.yml ${str} build "${@:2}"
    fi
    ;;
  start)
    if [[ -z "$2" ]] ; then
        echo "Starting system with core modules:"
        echo $modules_up
        docker compose -f docker-compose-base.yml ${str} build ${modules_up}
        docker compose -f docker-compose-base.yml ${str} up ${modules_up}
    else
        echo "Starting specific modules: "+"${@:2}"
        docker compose -f docker-compose-base.yml ${str} build "${@:2}"
        docker compose -f docker-compose-base.yml ${str} up "${@:2}"
    fi
    ;;
  stop)
    echo "Taking down full system..."
    docker compose -f docker-compose-base.yml ${str} down
    ;;
  clean)
    echo "Taking down full system and removing volume data..."
    docker compose -f docker-compose-base.yml ${str} down -v
    ;;
  config)
    echo "Showing the merged compose file that will be used..."
    docker compose -f docker-compose-base.yml ${str} config
    ;;
  list)
    echo "Listing all available modules..."
    for module in ${modules[@]}; do
            echo - ${module%%:*}" "
    done
    ;;
  auto-completion)
    the_source=$(readlink -f -- "${BASH_SOURCE[0]}")
    the_dirname=$(dirname "${the_source}")
    the_filename=$(basename "${the_source}")
    echo $'\n'complete -W \"\$"(ls -C ${the_dirname}/modules/)"\" ./${the_filename} >> ~/.bashrc
    . ~/.bashrc
    echo "Added auto-completion to .bashrc"
    ;;
  *)
    echo "Please use either 'build', 'start', 'stop', 'clean', 'config', 'list', 'auto-completion', or call without args to generate the necessary configuration without doing anything else."
    echo "Add space-separated module names afterwards to perform the operation for specific modules."
    ;;
esac
