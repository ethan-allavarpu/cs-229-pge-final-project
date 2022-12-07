# Inspiration and help from https://stackoverflow.com/questions/8751497/latitude-longitude-coordinates-to-state-code-in-r/8751965#8751965
library(maps)
library(maptools)
library(sp)
library(tidyverse)

# Extract county from latitude and longitude ----
get_county <- function(long_lat) {
  # County mappings
  counties <- maps::map(
    "county", fill = TRUE, col = "transparent", plot = FALSE
  )
  # Split county IDs
  county_ids <- sapply(strsplit(counties$names, ":"), function(x) x[1])
  # Polygon mappings of counties
  county_sp <- maptools::map2SpatialPolygons(
    counties, IDs = county_ids,
    proj4string = CRS("+proj=longlat +datum=WGS84")
  )
  spatial_points <- sp::SpatialPoints(
    long_lat, proj4string = CRS("+proj=longlat +datum=WGS84")
  )
  # See where each point lies (in what county)
  indices <- over(spatial_points, county_sp)
  county_names <- sapply(county_sp@polygons, function(x) x@ID)
  county_names[indices]
}

extract_county <- function(state_county) {
  if (is.na(state_county)) {
    return(state_county)
  }
  stringr::str_split(string = state_county, pattern = ",")[[1]][2]
}

train_x <- readr::read_csv("data/processed/x_train.csv", col_select = -1)
train_y <- readr::read_csv("data/processed/y_train.csv", col_select = -1)

long_lat <- train_x %>% select(longitude, latitude)
counties <- get_county(data.frame(long_lat)) %>%
  # Get county from each value
  purrr::map(extract_county) %>%
  unlist()

# Total Amount of Shutoff Time (minutes) by County ----
ca_counties <- map_data("county") %>% filter(region == "california")
# Calculate total time, grouped by county
county_shutoff <- dplyr::bind_cols("county" = counties, train_y) %>%
  group_by(county) %>%
  summarise(stat = sum(time_out_min))

ca_info <- left_join(ca_counties, county_shutoff,
                     by = c("subregion" = "county"))

shutoff_plot <- ca_info %>%
  # Grid to plot
  ggplot(aes(x = long, y = lat, group = group, fill = stat)) +
  # County borders
  geom_polygon(color = "black") +
  # Color based on total shutoff time
  scale_fill_gradient(low = rgb(0.5, 0, 0, alpha = 0),
                      high = rgb(0.33, 0, 0),
                      limits = c(0, max(county_shutoff$stat, na.rm = TRUE)),
                      na.value = "white",
                      name = "Total Shutoff Time\n(Minutes)") +
  labs(title = "Total Shutoff Time (Minutes) for PSPS Events by County",
       x = "Longitude", y = "Latitude") +
  theme_minimal() +
  # Adjust text sizing
  theme(plot.title = element_text(size = 16),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12)) +
  # 1:1 aspect ratio
  coord_fixed()

# Save PDF file
pdf("visuals/shutoff-by-county.pdf", width = 7, height = 5)
invisible(print(shutoff_plot))
dev.off()
