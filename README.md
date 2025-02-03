# Kilosort sorting components for SpikeInterface

This is a friendly port of some [Kilosort](https://github.com/MouseLand/Kilosort/) parts to be adapted in the SpikeInterface sortingcomponents framework.

At the moment only the template matching component is ported.

The initial plan was to integrate directly into SpikeInterface these 40 lines of code, but due to Kilosort
license (GPL 3), we need an intermediate package with the same license.

Note that the template-matching peeler is not a copy/paste from the [Kilosort implementation]
(https://github.com/MouseLand/Kilosort/blob/main/kilosort/template_matching.py),
but rather an adaptation to the SpikeInterface framework. Only a portion of the code is from the original file.

