#!/usr/bin/env python
import os
import wx

from ConfigParser import ConfigParser

from dmcap.controller import DMCController

if __name__ == '__main__':
    app = wx.App(False)

    # Looking for a configuration file
    cfg_locations = [os.path.expanduser("~") + '/.3dmcap/3dmcap.cfg',
                     '/etc/3dmcap.cfg'
                     '/usr/local/share/3dmcap/3dmcap.cfg',
                     '/usr/share/3dmcap/3dmcap.cfg']
    cfg_path = None
    for cp in cfg_locations:
        if os.access(cp, os.F_OK):
            cfg_path = cp
            break

    config = ConfigParser()
    config.read(cfg_path)

    control = DMCController(config)
    app.MainLoop()
