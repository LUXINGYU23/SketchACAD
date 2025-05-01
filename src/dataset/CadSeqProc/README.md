This folder contains codes for processing the DeepCAD Jsons.


### 1. Json to Vector Representation

```python
import json
from cad_sequence import CADSequence
from utility.macro import MAX_CAD_SEQ_LEN, N_BIT

json_path="./data/cad_json/0000/00000007.json"
with open(json_path,"r") as f:
    json_data=json.load(f)

cad_seq, cad_vec, flag_vec, index_vec=CADSequence.json_to_vec(data=json_data,bit=N_BIT,padding=True,max_cad_seq_len=MAX_CAD_SEQ_LEN)
```

### 2. Generate Mesh/Brep/Point Cloud from Json/Vec

_NOTE: Use `CADSequence.from_vec(vec, denumericalize=True)` to create the class if your input is a vector representation._

_WARNING: Do not generate the CAD Model while the class contains quantized parameters. Use the method `.denumericalize(bit=N_BIT)` during that case._

```python
import json
from cad_sequence import CADSequence
from utility.macro import MAX_CAD_SEQ_LEN, N_BIT

json_path="./data/cad_json/0000/00000007.json"
with open(json_path,"r") as f:
    json_data=json.load(f)

cad_seq=CADSequence.json_to_NormalizedCAD(data=json_data, bit=N_BIT)

# <!-- ------------------------ Generate Brep or Mesh ------------------------ -->
# NOTE: It will be saved in os.path.join(output_dir,filename+".step")

brep=cad_seq.save_stp(filename=filename, output_dir=output_dir, type="step") # type="stl" for mesh

# <!-- ------------------------ Generate Point Cloud ------------------------- -->
# NOTE: filename without .ply
brep=cad_seq.save_points(filename=filename, output_dir=output_dir, n_points=8192, pointype="uniform")

```

### 3. Generate Mesh/Brep/Point Cloud from Minimal Json

```python
import json
from cad_sequence import CADSequence
from utility.macro import MAX_CAD_SEQ_LEN, N_BIT

json_path="./data/cad_json/0000/minimal_json.json"
with open(json_path,"r") as f:
    json_data=json.load(f)

cad_seq=CADSequence.from_minimal_json(json_data)

# <!-- ------------------------ Generate Brep or Mesh ------------------------ -->
# NOTE: It will be saved in os.path.join(output_dir,filename+".step")

brep=cad_seq.save_stp(filename=filename, output_dir=output_dir, type="step") # type="stl" for mesh

# <!-- ------------------------ Generate Point Cloud ------------------------- -->
# NOTE: filename without .ply
brep=cad_seq.save_points(filename=filename, output_dir=output_dir, n_points=8192, pointype="uniform")

```

### 4. Revolve Operation Support

The library now supports revolve operations in addition to extrusion. Revolve features allow for the creation of rotational geometry by rotating a profile around a specified axis.

```python
import json
from cad_sequence import CADSequence
from utility.macro import N_BIT

json_path = "./data/cad_json/0000/revolve_example.json"
with open(json_path, "r") as f:
    json_data = json.load(f)

# Convert JSON to CADSequence with revolve operation support
cad_seq = CADSequence.json_to_NormalizedCAD(data=json_data, bit=N_BIT)

# Create a model with revolve operations
cad_seq.create_cad_model()

# Save the model with revolve operations
cad_seq.save_stp(filename="revolve_model", output_dir=output_dir, type="step")
```

When converting between formats, revolve operations are properly handled in the vector representation and can be correctly restored:

```python
# Convert to vector representation
cad_seq, cad_vec, flag_vec, index_vec = CADSequence.json_to_vec(data=json_data, bit=N_BIT, padding=True)

# Restore from vector representation
restored_cad_seq = CADSequence.from_vec(cad_vec, bit=N_BIT, post_processing=True, denumericalize=True)

# Create and save the restored model with revolve operations
restored_cad_seq.create_cad_model()
restored_cad_seq.save_stp(filename="restored_revolve", output_dir=output_dir, type="step")
```

### 5. To Check the parameters of the CAD Model

```python
print(cad_seq)
```

```python
# Output
CAD Sequence:

    - Sketch:
       - CoordinateSystem:
            - Rotation Matrix [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            - Translation [0. 0. 0.]
       - Face:
          - Loop: Start Point: [0.0, 0.0], Direction: collinear
              - Circle: center([0.3765 0.3765]),             radius(0.3765), pt1 [0.37647059 0.75      ]


    - ExtrudeSequence: (profile_uids: [['FQgWGf8WhgalpUy', 'JGC']], extent_one: 0.1015625, extent_two: 0.0, boolean: 0, sketch_size: 0.75) Euler Angles [0. 0. 0.]

    - RevolveSequence: (profile_uids: [['F9847h_2', 'K34']], angle: 360.0, axis_start: [0.0, 0.375, 0.375], axis_end: [1.0, 0.375, 0.375], boolean: 0) Euler Angles [0. 0. 0.]
```

### 6. Testing Revolve Operations

You can use the test suite to verify that revolve operations are working correctly:

```python
# Test JSON to vector to STEP conversion with revolve operations
python test_json2vec2json2step.py --input /path/to/revolve_examples --output ./test_results
```

This script will:
1. Convert JSON files to vector representation
2. Restore models from vectors
3. Generate STEP files from both original and restored models
4. Render multiple views of the models for visual comparison