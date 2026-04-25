#include "StyleGAN.h"

StyleGANLayer::StyleGANLayer() {
}

void StyleGANLayer::Init(int stride, const StyleGANParams& params) {
    this->stride = stride;
    this->stylegan_params = params;

    input_width = stylegan_params.target_resolution;
    input_height = stylegan_params.target_resolution;
    input_depth = 3;

    int latent_size = stylegan_params.latent_dim;

    String disc_t =	"[\n"
                    "\t{ \"type\" : \"input\", \"input_width\":" + FormatInt(input_width) + ", \"input_height\":" + FormatInt(input_height) + ", \"input_depth\":" + FormatInt(input_depth) + "},\n"
                    "\t{ \"type\" : \"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":64, \"activation\":\"relu\"},\n"
                    "\t{ \"type\" : \"conv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":128, \"activation\":\"relu\"},\n"
                    "\t{ \"type\" : \"fc\", \"neuron_count\":1, \"activation\":\"sigmoid\"},\n"
                    "\t{ \"type\" : \"adam\", \"learning_rate\":" + FormatDouble(stylegan_params.learning_rate) + ", \"beta1\":0.0, \"beta2\":0.99, \"batch_size\":16, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!disc.MakeLayers(disc_t))
        throw Exc("Discriminator network loading failed");

    String gen_t =	"[\n"
                    "\t{ \"type\" : \"input\", \"input_width\":" + FormatInt(latent_size) + ", \"input_height\":1, \"input_depth\":1},\n"
                    "\t{ \"type\" : \"fc\", \"neuron_count\":4*4*512, \"activation\":\"relu\"},\n"
                    "\t{ \"type\" : \"unflatten\", \"width\":4, \"height\":4, \"depth\":512},\n"
                    "\t{ \"type\" : \"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":256, \"activation\":\"relu\"},\n"
                    "\t{ \"type\" : \"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":128, \"activation\":\"relu\"},\n"
                    "\t{ \"type\" : \"deconv\", \"sx\":4, \"sy\":4, \"stride\":2, \"pad\":1, \"filters\":3, \"activation\":\"tanh\"},\n"
                    "\t{ \"type\" : \"adam\", \"learning_rate\":" + FormatDouble(stylegan_params.learning_rate) + ", \"beta1\":0.0, \"beta2\":0.99, \"batch_size\":16, \"l2_decay\":0.0001}\n"
                    "]\n";

    if (!gen.MakeLayers(gen_t))
        throw Exc("Generator network loading failed");
}

void StyleGANLayer::Train() {
    Net& gen_net = gen.GetNetwork();
    Net& disc_net = disc.GetNetwork();

    SessionData& data = disc.Data();
    if (data.GetDataCount() > 0) {
        int real_idx = Random(data.GetDataCount());
        const Vector<double>& real_image_vec_ref = data.Get(real_idx);
        Vector<double> real_image_vec = clone(real_image_vec_ref);
        Volume real_image;
        real_image.Set(input_width, input_height, input_depth, real_image_vec);

        disc_net.Forward(real_image, true);
        Vector<double> real_target(1); real_target[0] = 1.0;
        disc_cost_av.Add(disc_net.Backward(real_target));

        Volume noise; noise.Init(stylegan_params.latent_dim, 1, 1);
        for (int i = 0; i < stylegan_params.latent_dim; i++) noise.Set(i, 0, 0, 2.0 * Randomf() - 1.0);
        Volume& fake_image = gen_net.Forward(noise, false);

        disc_net.Forward(fake_image, true);
        Vector<double> fake_target(1); fake_target[0] = 0.0;
        disc_cost_av.Add(disc_net.Backward(fake_target));

        disc.GetTrainer().TrainImplem();
    }

    Volume noise; noise.Init(stylegan_params.latent_dim, 1, 1);
    for (int i = 0; i < stylegan_params.latent_dim; i++) noise.Set(i, 0, 0, 2.0 * Randomf() - 1.0);

    Volume& fake_image_gen = gen_net.Forward(noise, true);
    disc_net.Forward(fake_image_gen, true);

    Vector<double> gen_target(1); gen_target[0] = 1.0;
    gen_cost_av.Add(disc_net.Backward(gen_target));
    
    gen.GetTrainer().TrainImplem();
}

void StyleGANLayer::SampleInput() {}
void StyleGANLayer::SampleOutput() {}

Volume& StyleGANLayer::Generate(Volume& input) {
    return gen.GetNetwork().Forward(input, false);
}
