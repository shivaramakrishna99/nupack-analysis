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
from latch.resources.conditional import create_conditional_section

class Material(Enum):
    dna = "DNA"
    rna = "RNA"
    rna95 = "RNA95"

class Ensemble(Enum):
    stacking = "stacking"
    nostacking = "nostacking"

@small_task
def nupack_analysis(
    strands_file: LatchFile, 
    concentrations: LatchFile,
    material: str = Material.rna,
    ensemble: str = Ensemble.stacking,
    temperature: float = 37.0,
    sodium: float = 1.0,
    magnesium: float = 0.0,
    out: str = "nupack-analysis",
) -> LatchFile:

    nt_model = Model(material=material, ensemble=ensemble, celsius=temperature, sodium=sodium, magnesium=magnesium)

    with open(strands_file, "r") as f:
        strands = f.read().splitlines()
    f.close()

    strand_names = []
    strand_sequences = []

    for l in strands:
        if l.startswith('>'):
            l = l.replace('>','').strip()
            strand_names.append(l)
    
    outFile = f"/{out}.txt"

    content = f"NUPACK ANALYSIS RESULTS\n{strand_names}"
    with open(outFile, "w") as f:
        f.write(content)
    
    return LatchFile(outFile, f"latch:///{outFile}")

metadata = LatchMetadata(
    display_name="NUPACK Complex Analyis",
    documentation="https://docs.nupack.org",
    author=LatchAuthor(
        name="NUPACK Team",
        email="support@nupack.org",
        github="https://github.com/beliveau-lab/NUPACK",
    ),
    repository="https://github.com/author/my_workflow",
    license="BSD-3-Clause",
)

metadata.parameters["strands_file"] = LatchParameter(
    display_name="FASTA File",
    description="Upload FASTA file containing strands",
    hidden=False,
    section_title="Input",
)
metadata.parameters["concentrations"] = LatchParameter(
    display_name="Concentrations File",
    description="Upload list of concentrations",
)

@workflow(metadata)
def analysisNUPACK(
    strands_file: LatchFile, 
    concentrations: LatchFile,
    material: Material = Material.rna,
    ensemble: Ensemble = Ensemble.stacking,
    temperature: float = 37.0,
    sodium: float = 1.0,
    magnesium: float = 0.0,
    out: str = "tube-analysis",
) -> LatchFile:

    """NUPACK Utilities for a complex formed between two nucleotide strands

    # NUPACK - Utility Algorithms and Calculations
    ---

    ## **How to use**
    ---

    1. Specify the model by choosing the nucleic acid type. This parameter is based on parameter sets obtained from different research papers. Check them out [here](https://docs.nupack.org/model/#material)
    
    2. Specify the sequences of the first and second strand, followed by the structure of the complex in the dot bracket notation.

    3. Select the utility functions that require sequence information only.

    4. Select the utility functions that require sequence and structure information.

    5. Specify any other changes in the construction of the Model() object using the hidden parameters such as ensemble type and ion concentrations. 

    6. Run the workflow!

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
    
    Args:
        material:
            __metadata__:
                display_name: "Nucleic Acid Type"
                _tmp:
                    section_title: "Sequence and Structure Details"
                appearance:
                    comment: "Choose between DNA and RNA free energy parameter sets. Default is 'rna', based on Matthews et al., 1999"
        ensemble:
            __metadata__:
                display_name: "Ensemble Type"
                _tmp:
                    section_title: Additional Model Specification
                    hidden: true
                appearance:
                    comment: "Choose between stacking and non stacking ensemble states. Default is set to 'stacking'."
        temperature:
            __metadata__:
                display_name: "Temperature (in degree Celsius)"
                _tmp:
                    hidden: true
                appearance:
                    comment: "Temperature of system. Default is 37 °C"
        sodium:
            __metadata__:
                display_name: "Na+ concentration (in M)"
                _tmp:
                    hidden: true
                appearance:
                    comment: "The sum of the concentrations of (monovalent) sodium, potassium, and ammonium ions, is specified in units of molar. Default: 1.0, Range: [0.05,1.1]"
        magnesium:
            __metadata__:
                display_name: "Mg++ (in nM). Default is 0 nM"
                _tmp:
                    hidden: true
                appearance:
                    comment: "The concentration of (divalent) magnesium ions, is specified in units of molar. Default: 0.0, Range: [0.0,0.2]"
        out:
            __metadata__:
                display_name: "Output File Name"
                _tmp:
                    section_title: Output
                appearance:
                    comment: "Name your file containing results from chosen NUPACK Utility Programs"            
    """
    
    return nupack_analysis(
    strands_file=strands_file,
    concentrations=concentrations,
    material=material,
    ensemble=ensemble,
    temperature=temperature,
    sodium=sodium,
    magnesium=magnesium,
    out=out
    )
