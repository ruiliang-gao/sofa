#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess

# Simple command
out = subprocess.call(['history'], shell=True)
print("OUT"+str(out))
