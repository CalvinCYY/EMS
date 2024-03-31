# EMS - Efficient Molecular Storage
Mol_translator lite

The purpose of this code is a full refactoring of the mol_translator codebase, many functions were redundant or bloated with older syntax. At this point it is much easier to do a complete rewrite. End goal is to essentially pass in the folder with some other parameters e.g. index_position to find the ID, and smiles strings checks. Makes it easier to error check if all processing is ran for no user input for end user.

v0.0.1
SDF parsing included - Functionally goes end to end from nmredata.sdf to atoms and pairs dataframe.
