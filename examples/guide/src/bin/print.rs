use guide::{
    backend::{create_device, GuideBackend},
    model::ModelConfig,
};

fn main() {
    let device = create_device();
    let model = ModelConfig::new(10, 512).init::<GuideBackend>(&device);

    println!("{model}");
}
