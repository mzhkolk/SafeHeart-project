class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.25):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,  stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels) 
        self.activation1 = nn.ELU(inplace=True)  
        self.activation2 = nn.ELU(inplace=True)  
        self.downsample_block = None
        if stride != 1 or in_channels != out_channels:
            self.downsample_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.activation1(out)
        out = self.dropout(out)  
        out = self.bn2(self.conv2(out))

        if self.downsample_block is not None:
            residual = self.downsample_block(residual)

        out += residual
        out = self.activation2(out)
        out = self.dropout(out)  

        return out


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        s = 16
        
        kernel_size = 3
        encoder_layers = []
        encoder_layers.append(nn.Conv2d(1, 8*s, kernel_size=kernel_size, stride=1, padding=1))
        encoder_layers.append(nn.BatchNorm2d(8*s))
        encoder_layers.append(nn.ELU(inplace=True))

        encoder_layers.append(ResidualBlock(8*s, 8*s))
        encoder_layers.append(nn.Conv2d(8*s, 16*s, kernel_size=kernel_size, stride=1, padding=1))
        encoder_layers.append(nn.BatchNorm2d(16*s))
        encoder_layers.append(nn.ELU(inplace=True))
        encoder_layers.append(ResidualBlock(16*s, 16*s))


        self.encoder = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 13, 28)
            encoder_output_shape, decoder_input_shape = self._get_encoder_output_shape(dummy_input)
        print(encoder_output_shape)
        self.fc1 = nn.Linear(encoder_output_shape, 128)

        self.fc_mu = nn.Linear(encoder_output_shape, self.latent_dim)
        self.fc_log_var = nn.Linear(encoder_output_shape, self.latent_dim)

        decoder_layers = []

        self.decoder_input = nn.Linear(self.latent_dim, encoder_output_shape)
        self.decoder_reshape = decoder_input_shape 
        
        decoder_layers.append(ResidualBlock(16*s, 16 *s))
        decoder_layers.append(nn.ConvTranspose2d(16*s, 8*s, kernel_size=kernel_size, stride=1, padding=1))
        decoder_layers.append(nn.BatchNorm2d(8*s))
        decoder_layers.append(nn.ELU(inplace=True))
        
        decoder_layers.append(ResidualBlock(8*s, 8*s))
        decoder_layers.append(nn.ConvTranspose2d(8*s, 1, kernel_size=kernel_size, stride=1, padding=1))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def decode(self, z):
        x = self.decoder_input(z)
        batch_size = x.size(0)
        decoder_reshape_without_batch = self.decoder_reshape[1:]
        x = x.view(batch_size, *decoder_reshape_without_batch)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, z, mu, log_var
    
    def _get_encoder_output_shape(self, x):
        x = self.encoder(x)
        encoder_output_shape = x.view(x.size(0), -1).shape[1]
        decoder_input_shape = x.size()
        return encoder_output_shape, decoder_input_shape


class VAE_Loss(nn.Module):
    def __init__(self, beta):
        super(VAE_Loss, self).__init__()
        self.beta = beta

    def forward(self, decoded, x, mu, log_var):
        reconstruction_loss= F.mse_loss(decoded, x, reduction='mean') #/ decoded.size(0)
        kl_divergence_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss =  reconstruction_loss + self.beta * kl_divergence_loss
        return total_loss, reconstruction_loss, self.beta *kl_divergence_loss    
