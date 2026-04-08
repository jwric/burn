#![recursion_limit = "131"]
use burn::data::dataset::Dataset;
use guide::{
    backend::{GuideBackend, create_device},
    inference,
};

fn main() {
    let device = create_device();

    // All the training artifacts are saved in this directory
    let artifact_dir = "/tmp/guide";

    // Infer the model
    inference::infer::<GuideBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
