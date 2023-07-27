# A Detailed Evaluation Methodology for Revision Policies in Incremental Sequence Labelling

Code to accompany the manuscript: The Road to Quality is Paved with Good 
Revisions: A Detailed Evaluation Methodology for Revision Policies in 
Incremental Sequence Labelling (Brielen Madureira, Patrick Kahardipraja and 
David Schlangen, SIGdial 2023).

## Files

- ```setup.sh```: A bash script that create the conda environment and the needed empty directories.
- ```inceval/aux.py```: Defines the symbol used to fill the unused upper part 
of the incremental charts, which are represented as numpy arrays. Defines enum
variables for the silver and gold standard.
- ```inceval/edit.py```: Classes that represent edits.
- ```inceval/revision.py```: Classes that represent revisions.
- ```incoutputs.py```: Class to represent an incremental chart. Contains all
the main metrics on sequence level.
- ```incdataset.py```: Class to represent a dataset of incremental charts.
Contains all the main metrics on dataset level.
- ```example.ipynb```: A Notebook with a demonstration of how to use the framework.
- ```analysis-tapir.ipynb```: The notebook used to generate results in the paper.
- ```characterisation.ipynb```: A notebook with more examples for each category.

## Set Up

Create the conda environment. Either run ```sh setup.sh``` for the exact steps 
of create via conda using ```conda env create -f environment.yml```. 
```sh setup.sh``` also creates one directory for the preprocessed data and one 
for the output figures. Put the incremental outputs into ```preprocessed/```.

## Replicating results

The plots and table in the paper have been generated using 
```analysis-tapir.ipynb```.

## Evaluating other models
The scripts in ```inceval``` implement all the metrics and characteristics 
described in the paper. See ```example.ipynb```` for a short demonstration of 
the main functionalities.

### Structures

The class ```IncOutputs``` represents the incremental chart of a sequence. It
can be used to compute metrics on sequence level, whereas ```IncData``` gets
a dictionary of ```IncOutputs``` and computed metrics on dataset level.

When ```IncOutputs``` is initialised, the used can define whether to use the
real 'true' labels or the final labels as gold standard (GOLD and SILVER enum
arguments, respectively). For models that perform revisions via recomputations,
the sequence of recomputation steps can also be passed as an argument, so that
additional metrics can be computed. It builds and makes accessible various
attributes: 

- ```chart```: a lower triangular matrix containing all the output prefixes
- ```edits```: a lower triangular matrix containing 1 when an edit occurred
- ```edit_qualities```: a lower triangular matrix containing all the charactesised
- ```revision_qualities```: a sequence containing all the characterised revisions


They rely on the following objects:

- ```EditQualities```: a dataclass representing all the attributes of an edit.
- ```EditQualityChart```: a class representing the whole sequence of edits.
```self.chart``` contains a lower triangular matrix, filled with Nones or with
an ```EditQualities``` object in cells that represent edited labels.
- ```Revisionualities```: a dataclass representing all the attributes of a
revision.


### Usage

1. Build the incremental chart of a sequence represented with the 
```IncOutputs``` class. You can feed the full chart at once or incrementaly,
one prefix at a time.

2. Compute metrics on sequence level. Use the methods in ```IncOutputs```.

3. Compute metrics on dataset level. Create a dictionary that maps sequence
IDs to its IncOutputs object. This can be used ot initialise ```IncData```,
which allows dataset level metrics to be computed.

The notebook ```example.ipynb``` has a demonstration.

## Design decisions

- the first write is counted as occurring always upon a correct (empty) prefix
- the revision metrics are ```np.nan``` when no revision/addition occurred
- the recomputation metrics are ```np.nan``` when no recomputation is passed 
as an argument

## Testing

```bash
python -m unittest discover .
```

## TODOs

- implement corretion time score and correction time per token
- write unittests for a few extra methods not used in the paper and for incdata

## License

This code is licensed under the MIT License (see the ```LICENSE``` file).

## Citation

TBA
