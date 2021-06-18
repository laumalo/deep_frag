# General imports
import argparse as ap
import h5py
import torch
import numpy as np
import time
from rdkit import Chem
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Import DeepFrag tools
import sys
sys.path.append('./deepfrag')
sys.path.append('/usr/local/lib/python3.7/site-packages/')

from leadopt.model_conf import LeadoptModel, DIST_FN
from leadopt import grid_util, util
from leadopt.data_util import REC_TYPER, LIG_TYPER

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
    parser = ap.ArgumentParser(description=
                               "RANK A SET OF FRAGMENTS FOR A RECEPTOR/PARENT" +
                               "COMPLEX USING DEEPFRAG.")
    parser.add_argument("path_lib", type=str,
                        help="Path to the fragments library.")

    parser.add_argument("receptor_pdb", type=str,
                        help="Path to the receptor PDB.")

    parser.add_argument("parent_pdb", type=str,
                        help="Path to the parent PDB.")

    parser.add_argument("-g", "--growing", type=str,
                        help="Growing point for the fragment.")

    parsed_args = parser.parse_args()

    return parsed_args

def main(args):
    """
    It reads the command-line arguments and runs a prediction of the best
    fragments using the pre-trained CNN DeepFrag.
    It returns the ranking of the feched fragments from a library for a specific
    receptor-parent complex.

    Parameters
    ----------
    args : argparse.Namespace
        It contains the command-line arguments that are supplied by the user

    Examples
    --------
    From the command-line:
    >>> python prediction.py systems/DNA_ligase_core2_CH/fragments/fingerprints.h5 systems/DNA_ligase_core2_CH/receptor.pdb systems/DNA_ligase_core2_CH/parent.pdb -g C1
    """

    # DEEP_FRAG MODEL (pre-trained CNN parameters)
    TRAINED_MODEL = os.path.join(os.getcwd(),'data/final_model')

    # RECEPTOR
    rec_coords, rec_types = util.load_receptor_ob(args.receptor_pdb)

    # PARENT
    parent = Chem.MolFromPDBFile(args.parent_pdb)
    parent_coords = util.get_coords(parent)
    parent_types = np.array(util.get_types(parent)).reshape((-1,1))
    conn_idx = [atom.GetIdx() for atom in parent.GetAtoms()
                if atom.GetPDBResidueInfo().GetName().strip() == args.growing]
    conn = util.get_coords(parent)[conn_idx[0]]

    # Library of fragments (fingerprints)
    with h5py.File(args.path_lib, 'r') as f:
      f_smiles = f['smiles'][()]
      f_fingerprints = f['fingerprints'][()]

    USE_CPU = True
    device = torch.device('cpu') if USE_CPU else torch.device('cuda')

    # Load pre-trained model
    model = LeadoptModel.load(TRAINED_MODEL, device=device)

    # Prepare input (voxelization)
    batch = grid_util.get_raw_batch(
        rec_coords, rec_types, parent_coords, parent_types,
        rec_typer=REC_TYPER[model._args['rec_typer']],
        lig_typer=LIG_TYPER[model._args['lig_typer']],
        conn=conn,
        num_samples=32,
        width=model._args['grid_width'],
        res=model._args['grid_res'],
        point_radius=model._args['point_radius'],
        point_type=model._args['point_type'],
        acc_type=model._args['acc_type'],
        cpu=USE_CPU)
    batch = torch.as_tensor(batch)

    # Make prediction
    start = time.time()
    pred = model.predict(batch.float()).cpu().numpy()
    end = time.time()

    print('Generated prediction in %0.3f seconds' % (end - start))

    # Compute the average fingerprint.
    avg_fp = np.mean(pred, axis=0)

    # Grab the model distance function (cosine similarity).
    dist_fn = DIST_FN[model._args['dist_fn']]

    # The distance functions are implemented in pytorch so we need to convert
    # our numpy arrays to a torch Tensor.
    dist = dist_fn(
        torch.Tensor(avg_fp).unsqueeze(0),
        torch.Tensor(f_fingerprints))

    # Pair smiles strings and distances.
    dist = list(dist.numpy())
    scores = list(zip(f_smiles, dist))
    scores = sorted(scores, key=lambda x:x[1])

    # Fragments ranking
    output_ranking = os.path.join(os.path.dirname(args.path_lib), 'ranking.txt')
    f = open(output_ranking, 'w')
    print('Fragments ranking:')
    for idx, x in enumerate(scores):
        smi, score = x
        print(idx + 1 , score, smi.decode('ascii'))
        f.write('{} {} {} {}'.format(idx + 1 ,score, smi.decode('ascii'), '\n'))

if __name__ == '__main__':
    args = parse_args()
    main(args)
