"""Synthetic fixtures test the archived aggregation utility, not historical runs."""
from pathlib import Path
import json, tempfile, unittest
from verify import load_module, recompute, integrity

P=load_module('archived_parser','project/build_results_json_from_runs.py')
C=load_module('archived_ratios','project/compute_speedup.py')

class RecordTests(unittest.TestCase):
    def test_duration_units(self):
        self.assertEqual(P.parse_duration_to_seconds('1h2m3.125s'),3723.125)
        self.assertEqual(P.parse_duration_to_seconds('4m30.536s'),270.536)
        self.assertIsNone(P.parse_duration_to_seconds('invalid'))

    def test_last_real_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'output.log';p.write_text('real\t1m2s\nnoise\nreal\t1m3.5s\n')
            self.assertEqual(P.read_real_seconds(p),63.5)

    def test_missing_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):P.read_real_seconds(Path(tmp)/'missing.log')

    def test_max_min_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots=[]
            for i,seconds in enumerate([10,30,20]):
                root=Path(tmp)/str(i);q=root/'scene/case1';q.mkdir(parents=True)
                (q/'output.log').write_text(f'real\t{seconds}s\n');roots.append(root)
            self.assertEqual(P.pick_time_across_roots(roots,'scene',1,'min')[0],10)
            self.assertEqual(P.pick_time_across_roots(roots,'scene',1,'max')[0],30)

    def test_psnr_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);q=root/'scene/case1';q.mkdir(parents=True)
            (q/'psnr_result.json').write_text(json.dumps({'psnr':31.25}))
            self.assertEqual(P.read_psnr(root,'scene',1),31.25)
            self.assertIsNone(P.read_psnr(root,'scene',2))

    def test_missing_and_zero_time(self):
        x=C.compute_speedup({'s':{'c':{'seconds':4}}},{'s':{'c':{'seconds':0}}})
        self.assertEqual(x['s']['c']['status'],'invalid_other_zero')
        x=C.compute_speedup({'s':{'c':{'seconds':4}}},{})
        self.assertEqual(x['s']['c']['status'],'missing')

    def test_archived_results(self):
        x=recompute();self.assertEqual(x['cases'],20)
        self.assertAlmostEqual(x['ratio_of_means'],3.714416668640218,places=12)

    def test_source_integrity(self):
        self.assertGreater(integrity()['verified_archive_files'],70)

if __name__=='__main__':unittest.main(verbosity=2)
