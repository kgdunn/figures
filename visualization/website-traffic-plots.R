web <- read.csv('http://openmv.net/file/website-traffic.csv')
summary(web)

# Plot the default boxplot: the days are not in the usual order
boxplot(web$Visits ~ web$DayOfWeek)

# Create a factor variable to reorder the days in a new order:
# The factor names MUST match the spelling used in the original file.
day.names <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
day.factor <- factor(web$DayOfWeek, level=day.names)

# Plot the boxplot in the new order:
bitmap('website-traffic-boxplots.jpg', res=300, pointsize=14, width=10, height=6)
par(mar=c(4, 4, 0.2, 0.2))  # (bottom, left, top, right) spacing around plot
boxplot(web$Visits ~ day.factor, ylab="Number of visits")
dev.off()

# Plot the data in a time-series plot: crude method - OK for now
plot(web$Visits, type="o")

# A better plot (not expected to get full grade for this question)
# Use the xts library; search the software tutorial for "xts" to see how.
library(xts)
date.order <- as.Date(web$MonthDay, format=" %B %d")
web.visits <- xts(web$Visits, order.by=date.order)
bitmap('website-traffic-timeseries.jpg', res=300, pointsize=14, width=10)
plot(web.visits, major.format="%b", ylab="Number of website visits")
dev.off()