import numpy as np
import h5py
import os
from rdkit import Chem
import argparse as ap

import sys
import logging
logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)


def parse_args():
    """
    It parses the command-line arguments.
    Parameters
    ----------
    args : list[str]
        List of command-line arguments to parse
    Returns
    -------
    parsed_args : argparse.Namespace
        It contains the command-line arguments that are supplied by the user
    """
    parser = ap.ArgumentParser(description="ADDS CUSTOM FRAGMENTS TO THE SET.")

    parser.add_argument("path", type=str,
                        help="Path to folder containing the fragments.")

    parser.add_argument("--only", dest="only", action='store_true',
                        help="Generate a library only with the fetched " +
                        "fragments. Default: False")

    parser.add_argument("-o","--output", type=str, default='fingerprints.h5',
                        help="Output file name. Default: fingerprints.h5")

    parser.set_defaults(only=False)
    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    """
    It reads the command-line arguments and generates a fingerprint library with
    the input fragments.

    It returns a .h5 file with the fingerprints and the corresponding SMILES.

    Parameters
    ----------
    args : argparse.Namespace
        It contains the command-line arguments that are supplied by the user

    Examples
    --------
    From the command-line:
    >>> python generate_library.py systems/DNA_ligase_core2_CH/fragments
    """

    # DeepFrag fingerprints library
    PATH_DEEPFRAG_FRAGS = os.path.join(os.getcwd(),'data/fingerprints.h5')

    # Output path
    OUTPUT_PATH_NEW_LIB = os.path.join(args.path, args.output)

    # Path to the folder with the fragments to add to the library
    FRAGMENTS_FILES = [f for f in os.listdir(args.path)
                       if f.endswith('.pdb')]
    try:
        CONF_FILE = os.path.join(args.path, [f for f in os.listdir(args.path)
                                             if f.endswith('.conf')][0])

        d = {}
        with open(CONF_FILE, 'r') as f:
            data = f.readlines()
            for line in data:
                info = line.replace('\n', '').split()
                d[info[0]] = info[2]
    except Exception:
        raise ValueError('There has to be a configuration file in the fragments'
                         + ' folder with the growing point.')

    # Read the set of fragments from DeepFrag
    with h5py.File(PATH_DEEPFRAG_FRAGS, 'r') as f:
        f_smiles = f['smiles'][()]
        f_fingerprints = f['fingerprints'][()].astype(float)

    # Create empty variables for the new library
    if args.only:
        n_fingerprints = np.zeros((len(FRAGMENTS_FILES), 2048))
        n_smiles = []
    else:
        # Add to the existing library
        n_fingerprints = np.zeros(
            (len(f_fingerprints) + len(FRAGMENTS_FILES), 2048))
        n_fingerprints[0:len(f_fingerprints)] = f_fingerprints
        n_smiles = f_smiles

    # Loop for adding all the fragments from the input folder
    logging.info('  - {} fragments will be added into the library.'.format(
        len(FRAGMENTS_FILES)))

    for idx, fragment in enumerate(FRAGMENTS_FILES):
        logging.info('      - Adding {} to the fragment library.'.format(
            fragment))

        # Load molecule
        m = Chem.rdmolfiles.MolFromPDBFile(os.path.join(args.path, fragment),
                                           removeHs=False)

        # Fragment on H bond
        atom_name = d.get(fragment).strip()
        atom_id = [atom.GetIdx() for atom in m.GetAtoms()
                   if atom.GetPDBResidueInfo().GetName().strip() == atom_name][0]

        bonds_ids = []
        for bond in m.GetBonds():
            atoms = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            if atom_id in atoms:
                atom_elements = [m.GetAtomWithIdx(atoms[0]).GetSymbol(),
                                 m.GetAtomWithIdx(atoms[1]).GetSymbol()]
                if 'H' in atom_elements:
                    bonds_ids.append(bond.GetIdx())

        # Break on bond
        frag_prepared = Chem.FragmentOnBonds(m, [bonds_ids[0]])
        mols = Chem.GetMolFrags(frag_prepared, asMols=True, sanitizeFrags=True)
        for m in mols:
            atom_element = [a.GetSymbol() for a in m.GetAtoms()]
            if not atom_element == ['H', '*']:
                frag = m
                frag = Chem.RemoveHs(frag)

        # Generate SMILES
        new_smi = str(Chem.MolToSmiles(frag, isomericSmiles=False))

        # Generate fingerprint
        fp = Chem.rdmolops.RDKFingerprint(frag, maxPath=10)
        n_fp = list(map(int, list(fp.ToBitString())))

        # Check that the fingerprint is not already in the set (only if adding)
        if not args.only:
            for smi_set, fp_set in zip(f_smiles, f_fingerprints):
                    if (fp_set == n_fp).all():
                        logging.warning('          - The fragment {}'.format(new_smi) +
                                        ' is already in the library.')
                        break

            else:
                idx_out = idx if args.only else len(f_fingerprints) + idx
                n_smiles = np.append(n_smiles, new_smi)
                n_fingerprints[idx_out] = n_fp
                logging.info('          - Added {} to the fragment library.'.format(
                new_smi))

        else:
            n_smiles = np.append(n_smiles, new_smi)
            n_fingerprints[idx] = n_fp
            logging.info('          - Added {} to the fragment library.'.format(
            new_smi))

    # Output the fingerprint library file
    with h5py.File(OUTPUT_PATH_NEW_LIB, 'w') as f:
        f['fingerprints'] = n_fingerprints[:len(n_smiles)]
        f['smiles'] = np.array([smi.tobytes() for smi in n_smiles])
    logging.info('  - Output file saved as {}'.format(OUTPUT_PATH_NEW_LIB))


if __name__ == '__main__':
    args = parse_args()
    main(args)
