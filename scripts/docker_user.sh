#!/bin/bash

groupadd --gid "$DOCKER_GID" "$DOCKER_USER"
useradd -u "$DOCKER_UID" -g "$DOCKER_USER" -s /bin/bash "$DOCKER_USER"
usermod -a -G sudo "$DOCKER_USER"

echo "$DOCKER_USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/"$DOCKER_USER"
chmod 0440 /etc/sudoers.d/"$DOCKER_USER"

if [ "$*" = "" ]; then
    su "$DOCKER_USER"
else
    su "$DOCKER_USER" -c "$*"
fi
