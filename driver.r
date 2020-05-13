library(tensorflow)
library(keras)
library(imager)
library(tfdatasets)

data_dir = file.path('data-R')
classes= list.dirs(data_dir,full.names=FALSE,recursive=FALSE)
list_ds <- file_list_dataset(file_pattern = paste0(data_dir, "/*/*"))
# list_ds %>% reticulate::as_iterator() %>% reticulate::iter_next() 

get_label <- function(file_path) {
  parts <- tf$strings$split(file_path, "/")
  parts[-2] %>% 
    tf$equal(classes) %>% 
    tf$cast(dtype = tf$float32)
}

decode_img <- function(file_path, height = 128, width = 128) {
  
  size <- as.integer(c(height, width))
  
  file_path %>% 
    tf$io$read_file() %>% 
    tf$image$decode_jpeg(channels = 3) %>% 
    tf$image$convert_image_dtype(dtype = tf$float32) %>% 
    tf$image$resize(size = size)
}

preprocess_path <- function(file_path) {
  list(
    decode_img(file_path),
    get_label(file_path)
  )
}

# num_parallel_calls are going to be autotuned
labeled_ds <- list_ds %>% 
  dataset_map(preprocess_path, num_parallel_calls = tf$data$experimental$AUTOTUNE)

prepare <- function(ds, batch_size, shuffle_buffer_size) {
  
  if (shuffle_buffer_size > 0)
    ds <- ds %>% dataset_shuffle(shuffle_buffer_size)
  
  ds %>% 
    dataset_batch(batch_size) %>% 
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
}

# best arch - from GA
model <- keras_model_sequential() %>% 
  
  layer_conv_2d(filters = 64, kernel_size = c(6,6), activation = "elu",input_shape = c(128,128,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "elu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "elu") %>% 
  layer_flatten() %>% 
  layer_dense(units = 384, activation = "elu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(units = 384, activation = "elu") %>% 
  layer_dense(units = 2, activation = "softmax")

model %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = "adamax",
    metrics = "accuracy"
  )


history <- model %>% 
  fit(
    prepare(labeled_ds, batch_size = 10, shuffle_buffer_size = 1000),
    epochs = 5,
    verbose = 2
  )

plot(history)