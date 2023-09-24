from pathlib import Path
import shutil
from reimnerf.preprocessing.data import C3VD, remove_json_fames ,keep_only_json_fames
import argparse

def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument('sequences_dir', type=str)
    p.add_argument('output_dir', type=str)
    p.add_argument('--overwrite', action='store_true', help='delete old data and write new')
    p.add_argument('--sample_step', default=5, type=int, help='define the sample step to sample the initial dataset, default is 5')
    p.add_argument('--eval_step', default=5, type=int, help='every how many samples from the training set, to allocate for evaluation')
    return p

def main(args):

    # create the dst_path
    dataset_paths = list(Path(args.sequences_dir).iterdir())
    dst_path = Path(args.output_dir)

    
    for ds_p in dataset_paths:

        target_dir = dst_path/ds_p.name
        print(f'saved data under:{target_dir}')
        try:
            target_dir.mkdir(parents=True)
        except FileExistsError:
            if args.overwrite:
                shutil.rmtree(target_dir)
                target_dir.mkdir()
            else:
                print(f'skipping dataset:{ds_p.name} because already exists, manually delete it or specify the overwrite flag')
                continue
        c3vd_dataset = C3VD(ds_p)
        c3vd_dataset._normalize_dataset()

        c3vd_dataset.export_reim(target_dir, step=1, suffix='test', save_images=True)
        shutil.copy(target_dir/'transforms_test.json', target_dir/'transforms_train.json')
        shutil.copy(target_dir/'transforms_test.json', target_dir/'transforms_val.json')
        shutil.copy(target_dir/'transforms_test.json', target_dir/'transforms_true_test.json')


        keep_only_json_fames(target_dir/'transforms_train.json', keep_step=args.sample_step)
        keep_only_json_fames(target_dir/'transforms_val.json', keep_step=args.sample_step*args.eval_step)

        remove_json_fames(target_dir/'transforms_train.json', args.eval_step, skip_first=True)
        remove_json_fames(target_dir/'transforms_true_test.json', args.eval_step)
        remove_json_fames(target_dir/'transforms_val.json', 10000)
        c3vd_dataset.cleanup()

if __name__ == '__main__':
    parser = make_parser()
    main(parser.parse_args())