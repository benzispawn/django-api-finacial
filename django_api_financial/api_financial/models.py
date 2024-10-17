from django.db import models
from django.core.exceptions import ValidationError

# Create your models here.
class StockTimeline(models.Model):
    st_id = models.AutoField(primary_key=True)  # Auto-incrementing primary key for the timeline
    st_day = models.DateField()                 # Date in the format YYYY-MM-DD

    def __str__(self):
        return f"Stock Timeline {self.st_day}"

    def save(self, *args, **kwargs):
        # Check if a StockTimeline with the same st_day already exists
        if StockTimeline.objects.filter(st_day=self.st_day).exists():
            raise ValidationError(f"StockTimeline with st_day {self.st_day} already exists.")
        super().save(*args, **kwargs)

class StockData(models.Model):
    sd_id = models.AutoField(primary_key=True)  # Auto-incrementing primary key
    sd_open_price = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)  # Open price
    sd_close_price = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)  # Close price
    sd_high_price = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)  # High price
    sd_low_price = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True) # Low price
    sd_volume = models.DecimalField(max_digits=30, decimal_places=2, null=True, blank=True)  # Volume
    sd_price_prediction = models.DecimalField(max_digits=30, decimal_places=2, null=True, blank=True, default=None)
    sd_symbol = models.CharField(max_length=8)  # Stock symbol, varchar(8)
    st_id = models.ForeignKey(StockTimeline, on_delete=models.CASCADE)

    def __str__(self):
        return f"Stock Data {self.sd_id}"
