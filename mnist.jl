using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, crossentropy, throttle
using MLDatasets
using MLDataUtils
include("spikerate.jl")

function load_mnist_data(batch_size=64, train_split=0.8)
    # Load the MNIST dataset
    train_x, train_y = MNIST.traindata()
    test_x, test_y = MNIST.testdata()

    # Normalize the data
    train_x = Float32.(train_x)
    test_x = Float32.(test_x)

    # Reshape the data: MNIST images are 28x28, we add a singleton dimension
    train_x = reshape(train_x, 28, 28, 1, :)
    test_x = reshape(test_x, 28, 28, 1, :)

    # Convert labels to one-hot encoding
    train_y = onehotbatch(train_y, 0:9)
    test_y = onehotbatch(test_y, 0:9)

    # Split the training data into a training and validation set
    num_train = Int(floor(train_split * size(train_x, 4)))
    train_data = (train_x[:, :, :, 1:num_train], train_y[:, 1:num_train])
    val_data = (train_x[:, :, :, num_train+1:end], train_y[:, num_train+1:end])

    # Create data loaders
    train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=batch_size, shuffle=false)
    test_loader = DataLoader((test_x, test_y), batchsize=batch_size, shuffle=false)

    return train_loader, val_loader, test_loader
end

# Usage example
train_loader, val_loader, test_loader = load_mnist_data(1)

# get the first batch
x, y = first(train_loader)
# plot the first image
using GLMakie
fig = GLMakie.Figure()
ax = GLMakie.Axis(fig[1, 1])
GLMakie.heatmap!(ax, x[:, :, 1, 1], colormap=:viridis)
fig
# x is a 4D tensor of size (28, 28, 1, 64)
println(size(x))
# test x0
x0 = ones(10)*0.9
# convert it to the spike train
spike_train = spikerate.rate(x, 100)
println(size(spike_train))
# plot the spike train
using GLMakie
fig = GLMakie.Figure()
ax = GLMakie.Axis(fig[1, 1], xlabel="Time", ylabel="Spike")
GLMakie.heatmap!(ax, spike_train[1,:,:,1,1], colormap=:viridis)
# savefig("spike_train.png")