# MMEditing Viewer

## Introduction

**MMEditing Viewer** is a qualitative comparison tools to facilitate your research.

## Major features

- **Patch-based comparison**
  - Crop a patch on multiple images to compare
  - Batch comparison
  - Flexible settings on number of columns and size of images.
  - Save your comparison result
- **Before/After slider comparison**
  - Support both videos and images comparison
  - Record and save comparison results as a video clip

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

Please install opencv-pytho-headless version.

```shell
pip install opencv-python-headless
```

**Step 2.**
Install MMEditing.

```shell
git clone -b dev-1.x https://github.com/open-mmlab/mmediting.git
cd mmediting
```

**Step 3.**
Run

```shell
python tools/gui/gui.py
```

## Examples

## Contributing
