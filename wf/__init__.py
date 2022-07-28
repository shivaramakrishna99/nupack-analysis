"""
Carry out all NUPACK's Utilities functions
"""

from enum import Enum
from pathlib import Path
from latch import small_task, workflow
from natsort import as_ascii

from nupack import *  # Import NUPACK
import matplotlib.pyplot as plt
import pandas as pd
import re
from latch.types import LatchFile, LatchMetadata, LatchAuthor, LatchParameter, LatchAppearanceType, LatchRule


class Material(Enum):
    dna = "DNA"
    rna = "RNA"
    rna95 = "RNA95"


class Ensemble(Enum):
    stacking = "stacking"
    nostacking = "nostacking"


@small_task
def tubeAnalysis(
    strands_file: LatchFile,
    concentrations_file: LatchFile,
    max_size: int = 3,
    tube_name: str = "Tube 1",
    material: str = Material.rna,
    ensemble: str = Ensemble.stacking,
    temperature: float = 37.0,
    sodium: float = 1.0,
    magnesium: float = 0.0,
    out: str = "nupack-analysis",
) -> LatchFile:

    nt_model = Model(material=material, ensemble=ensemble,
                     celsius=temperature, sodium=sodium, magnesium=magnesium)

    with open(strands_file, "r") as f:
        strands = f.read().splitlines()
    f.close()

    with open(concentrations_file, "r") as f:
        concentrations = f.read().splitlines()
        concentrations = [float(i) for i in concentrations]
    f.close()

    strand_names = []
    strand_sequences = []

    for l in strands:
        if l.startswith('>'):
            l = l.replace('>', '').strip()
            strand_names.append(l)
        else:
            nt_match = re.match('^[ATGCUatgcu]', l)
            if nt_match != None:
                l = l.strip().upper()
                strand_sequences.append(l)

    strand_objs = []
    strands_dict = dict(zip(strand_names, strand_sequences))

    for name, sequence in strands_dict.items():
        strand_obj = Strand(sequence, name=name)
        strand_objs.append(strand_obj)

    tube_strands = dict(zip(strand_objs, concentrations))

    tube_obj = Tube(strands=tube_strands, complexes=SetSpec(
        max_size=max_size), name=tube_name)

    tube_results = tube_analysis(tubes=[tube_obj], model=nt_model)

    outFile = f"/{out}.txt"

    content = f"""NUPACK ANALYSIS RESULTS
    {tube_results}
    """
    with open(outFile, "w") as f:
        f.write(content)

    return LatchFile(outFile, f"latch:///{outFile}")


metadata = LatchMetadata(
    display_name="NUPACK - Tube Analysis",
    documentation="https://docs.nupack.org/analysis",
    author=LatchAuthor(
        name="NUPACK Team",
        email="support@nupack.org",
        github="https://github.com/beliveau-lab/NUPACK",
    ),
    repository="https://github.com/beliveau-lab/NUPACK",
    license="BSD-3-Clause",
)

metadata.parameters["strands_file"] = LatchParameter(
    display_name="FASTA File",
    description="File containing list of sequences in FASTA format",
    section_title="Input",
)
metadata.parameters["concentrations_file"] = LatchParameter(
    display_name="Concentrations File",
    description="File containing list of concentrations specified in float notation in the same order as the sequences in the FASTA file"
)
metadata.parameters["max_size"] = LatchParameter(
    display_name="Maximum Complex Size",
    description="Specify maximum number of interactions between strands",
    section_title="Tube Specifications",
)
metadata.parameters["tube_name"] = LatchParameter(
    display_name="Name of Tube",
    description="Provide a name for the tube"
)
metadata.parameters["material"] = LatchParameter(
    display_name="Nucleic Acid Type",
    description="Choose between DNA and RNA free energy parameter sets. Default is 'rna', based on Matthews et al., 1999",
    section_title="Model Specification",
    hidden=True,
)
metadata.parameters["ensemble"] = LatchParameter(
    display_name="Ensemble Type",
    description="Choose between stacking and non stacking ensemble states. Default is 'stacking'",
    hidden=True
)
metadata.parameters["temperature"] = LatchParameter(
    display_name="Temperature (in °C)",
    description="Temperature of system. Default: 37.0",
    hidden=True,
)
metadata.parameters["sodium"] = LatchParameter(
    display_name="Na+ (in M)",
    description="The total concentration of (monovalent) sodium, potassium, and ammonium ions, specified as molarity. Default: 1.0, Range: [0.05,1.1]",
    hidden=True,
    section_title="Additional Model Specification"
)
metadata.parameters["magnesium"] = LatchParameter(
    display_name="Mg++ (in nM)",
    description="The total concentration of (divalent) magnesium ions, specified as molarity. Default: 0.0, Range: [0.0,0.2]",
    hidden=True
)
metadata.parameters["out"] = LatchParameter(
    display_name="Output File Name",
    section_title="Output"
)


@workflow(metadata)
def tubeAnalysisNUPACK(
    strands_file: LatchFile,
    concentrations_file: LatchFile,
    max_size: int = 3,
    tube_name: str = "Tube 1",
    material: Material = Material.rna,
    ensemble: Ensemble = Ensemble.stacking,
    temperature: float = 37.0,
    sodium: float = 1.0,
    magnesium: float = 0.0,
    out: str = "tube-analysis",
) -> LatchFile:
    """Define and analyse a tube containing multiple nucleic acid strands

    # NUPACK - Analysis of Complexes in a Tube
    ---

    ## **How to use**
    ---

    1. Provide a FASTA file containing the names and sequences of the interacting strands in a tube
    
    2. Provide a .txt file containing a list of concentration values (in floating point notation) for each strand.  Limit one entry per line, written in floating point notation (eg: 1e-8) 

    2. Set a maximum complex size and give your tube analysis job a name

    3. Specify any other changes in the construction of the Model() object using the hidden parameters such as ensemble type and ion concentrations

    4. Run the workflow!

    ## **About**
    ---

    [NUPACK](https://docs.nupack.org/#about) is a growing software suite for the analysis and design of nucleic acid structures, devices, and systems serving the needs of researchers in the fields of nucleic acid nanotechnology, molecular programming, synthetic biology, and across the life sciences more broadly.

     ## **Citations**
    ---

    ### NUPACK Analysis Algorithms

    **Complex analysis and test tube analysis**

    - M.E. Fornace, N.J. Porubsky, and N.A. Pierce (2020). A unified dynamic programming framework for the analysis of interacting nucleic acid strands: enhanced models, scalability, and speed.  [ACS Synth Biol](https://pubs.acs.org/doi/abs/10.1021/acssynbio.9b00523) , 9:2665-2678, 2020. ( [pdf](http://www.nupack.org/downloads/serve_public_file/fornace20.pdf?type=pdf) ,  [supp info](http://www.nupack.org/downloads/serve_public_file/fornace20_supp.pdf?type=pdf) )

    - R. M. Dirks, J. S. Bois, J. M. Schaeffer, E. Winfree, and N. A. Pierce. Thermodynamic analysis of interacting nucleic acid strands.  [SIAM Rev](http://epubs.siam.org/doi/abs/10.1137/060651100) , 49:65-88, 2007. ( [pdf](http://www.nupack.org/downloads/serve_public_file/sirev07.pdf?type=pdf) )

    **Pseudoknot analysis**

    - R. M. Dirks and N. A. Pierce. An algorithm for computing nucleic acid base-pairing probabilities including pseudoknots.  [J Comput Chem](http://onlinelibrary.wiley.com/doi/10.1002/jcc.10296/abstract) , 25:1295-1304, 2004. ( [pdf](http://www.nupack.org/downloads/serve_public_file/jcc04.pdf?type=pdf) )

    - R. M. Dirks and N. A. Pierce. A partition function algorithm for nucleic acid secondary structure including pseudoknots.  [J Comput Chem](http://onlinelibrary.wiley.com/doi/10.1002/jcc.20057/abstract) , 24:1664-1677, 2003. ( [pdf](http://www.nupack.org/downloads/serve_public_file/jcc03.pdf?type=pdf) ,  [supp info](http://www.nupack.org/downloads/serve_public_file/jcc03_supp.pdf?type=pdf) )

    **Workflow Repository** - (https://github.com/shivaramakrishna99/nupack-utility-programs)

    **Acknowledgements** - (https://docs.nupack.org/#acknowledgments)

    *Authored by Shivaramakrishna Srinivasan. Feel free to reach out to me at shivaramakrishna.srinivasan@gmail.com*
    ---
    """

    return tubeAnalysis(
        strands_file=strands_file,
        concentrations_file=concentrations_file,
        max_size=max_size,
        tube_name=tube_name,
        material=material,
        ensemble=ensemble,
        temperature=temperature,
        sodium=sodium,
        magnesium=magnesium,
        out=out
    )
