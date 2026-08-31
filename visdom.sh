#!/bin/bash

docker run -d -p 8097:8097 --name visdom --restart unless-stopped hypnosapos/visdom:e0a912d

