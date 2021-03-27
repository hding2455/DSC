from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img_path', help='Image file path')
    parser.add_argument('save_path', help='Image save path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.7, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    import os
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    #image_names = ["000000433243","000000000776", "000000015497", "000000018193", "000000046497", "000000080274", "000000144300", "000000171757", "000000215723",
    #"000000080274", "000000095786", "000000170278", "000000367082", "000000452891", "000000459153", "000000489339", "000000550714", "000000564280"]
    #image_names = set(image_names)
    for img_file in os.listdir(args.img_path):
    #    if img_file[:-4] not in image_names:
    #        continue
    #    print(args.img_path + img_file)
        img_path = args.img_path + img_file
        save_path = args.save_path + img_file
        result = inference_detector(model, img_path)
        # show the results
        show_result_pyplot(model, save_path ,img_path, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
