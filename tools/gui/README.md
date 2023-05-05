# MMagic Viewer

- [Introduction](#introduction)
- [Major features](#major-features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Contributing](#contributing)

## Introduction

**MMagic Viewer** is a qualitative comparison tools to facilitate your research.

## Major features

- **Patch-based comparison**
  - Crop a patch on multiple images to compare
  - Batch comparison
  - Flexible settings on number of columns and size of images.
  - Save your comparison result
- **Before/After slider comparison**
  - Support both videos and images comparison
  - Record and save comparison results as a video clip

## Prerequisites

MMagic Viewer works on Linux, Windows and macOS. It requires:

- Python >= 3.6
- PyQt5
- opencv-python (headless version)

## Getting Started

**Step 0.**
Install PyQt5.

```shell
pip install PyQt5
```

**Step 1.**
Install and check opencv-python version.
If your meet following errors:

```
QObject::moveToThread: Current thread is not the object's thread.
Available platform plugins are: xcb... .
```

Please install opencv-python-headless version.

```shell
pip install opencv-python-headless
```

**Step 2.**
Install MMagic.

```shell
git clone https://github.com/open-mmlab/mmagic.git
```

**Step 3.**
Run

```shell
python tools/gui/gui.py
```

## Examples

**1. Patch-based comparison: batch images**

https://user-images.githubusercontent.com/49083766/199232588-7a07a3d9-725d-48be-89bf-1ffb45bd5d74.mp4

**2. Patch-based comparison: single image**

https://user-images.githubusercontent.com/49083766/199232606-f8539191-4bda-4b2c-975a-59020927abae.mp4

**3. Before/After slider comparison: images**

https://user-images.githubusercontent.com/49083766/199232615-2c56dcf1-0b42-41a5-884c-16a8f28a2647.mp4

**4. Before/After slider comparison: input video frames**

https://user-images.githubusercontent.com/49083766/199232617-e03a06dc-727b-43bb-8110-049d0fff28ba.mp4

**5. Before/After slider comparison: input Mp4 video**

https://user-images.githubusercontent.com/49083766/199232651-87d8064e-cbaf-4d30-b90b-94ee0af7d497.mp4

**6. Before/After slider comparison: record**

https://user-images.githubusercontent.com/49083766/199232634-eca70d28-8437-400a-8ab9-d2fe396b6ea9.mp4

## Contributing

We appreciate all contributions to improve MMagic Viewer. You can create your issue to report bugs or request new features. Welcome to give us suggestions or contribute your codes.
