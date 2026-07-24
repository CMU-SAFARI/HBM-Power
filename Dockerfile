FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        less \
        nano \
        ninja-build \
        openssh-client \
        pkg-config \
        python3 \
        python3-matplotlib \
        python3-numpy \
        python3-pandas \
        python3-pip \
        python3-seaborn \
        python3-venv \
        rsync \
        sudo \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=developer
ARG USER_UID=1001
ARG USER_GID=$USER_UID

RUN if ! getent group $USER_GID >/dev/null; then groupadd --gid $USER_GID $USERNAME; fi \
        && if getent passwd $USER_UID >/dev/null; then \
                existing_user="$(getent passwd $USER_UID | cut -d: -f1)"; \
                if [ "$existing_user" != "$USERNAME" ]; then \
                    usermod -l $USERNAME -d /home/$USERNAME -m "$existing_user"; \
                fi; \
            elif id -u $USERNAME >/dev/null 2>&1; then \
                usermod --uid $USER_UID --gid $USER_GID -d /home/$USERNAME -m -s /bin/bash $USERNAME; \
            else \
                useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash; \
            fi \
        && usermod --gid $USER_GID -s /bin/bash $USERNAME \
        && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
        && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /workspace
USER $USERNAME

CMD ["bash"]