#

This is a friendly port of some kilosort part to be adapted in the spikeinterface sortingcomponents framework.

At the moment only the template matching part is ported.

The initial plan was to integrate directly into spikeinterface this 40 lines of code but due to kilosort
licence (GPL 3), we need an intermediate package with the same licence.

Note that the peeler is not a copy/paste from 

https://github.com/MouseLand/Kilosort/blob/main/kilosort/template_matching.py

but an adaptation to the spikeinterface framework so only a portions of the code is from the original file.

