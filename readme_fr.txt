test1_fr usage:
1) create shortcut to the driver folder and update the variable foldername accordingly
2) run the script as is and it will recreate the histogram plots from results files (runtime<10sec)
3) how to get the file names in the LFW dataset:

-----------------------------------------------------------------------------------------------------
Variable name            | Size      | Description
-----------------------------------------------------------------------------------------------------
same_pairs_name_sets     | 10x300    | name of the person in the pairs of SAME person images
same_pairs_numbers_sets  | 10x300x2  | numbers of the images in the pairs of SAME person images
diff_pairs_names_sets    | 10x300x2  | names of the persons in the pairs of DIFFERENT persons images
diff_pairs_numbers_sets  | 10x300x2  | numbers of the images in the pairs of DIFFERENT person images
-----------------------------------------------------------------------------------------------------

    Construction of the file names of images of the same person:
image1_path = './lfw-py/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 0])
image2_path = './lfw-py/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(same_pairs_name_sets[set_cnt][npair], same_pairs_name_sets[set_cnt][npair], same_pairs_numbers_sets[set_cnt, npair, 1])

    Construction of the file names of images of two different persons:
image1_path = './lfw-py/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_names_sets[set_cnt][npair][0], diff_pairs_numbers_sets[set_cnt, npair, 0])
image2_path = './lfw-py/lfw_funneled/{:s}/{:s}_{:04d}.jpg'.format(diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_names_sets[set_cnt][npair][1], diff_pairs_numbers_sets[set_cnt, npair, 1])

where set_cnt is in [0..9] and npair is in [0..299]
