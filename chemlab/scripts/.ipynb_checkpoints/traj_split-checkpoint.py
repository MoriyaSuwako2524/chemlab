from chemlab.util.ml_data import MLData
test = MLData("/scratch/moriya/md/xtb_rhodamine/",type="xyz")
test.save_split(1000,200,400)
test.export_xyz_from_split("./split.npz",outdir="/scratch/moriya/md/rhodamine_tddft/xyz_splits/")