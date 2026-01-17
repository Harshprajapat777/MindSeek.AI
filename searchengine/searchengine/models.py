from django.db import models


class Expert(models.Model):
    uid = models.CharField(max_length=50, unique=True, primary_key=True)
    original_first_name = models.CharField(max_length=100, blank=True, null=True)
    original_surname = models.CharField(max_length=100, blank=True, null=True)
    english_first_name = models.CharField(max_length=100, blank=True, null=True)
    english_surname = models.CharField(max_length=100, blank=True, null=True)
    first_name = models.CharField(max_length=100)
    surname = models.CharField(max_length=100)
    email_address = models.EmailField(blank=True, null=True)
    primary_phone_number = models.CharField(max_length=50, blank=True, null=True)
    city = models.CharField(max_length=100, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    biography = models.TextField(blank=True, null=True)
    external_profile_link = models.URLField(max_length=500, blank=True, null=True)
    internal_biography = models.TextField(blank=True, null=True)
    released_profile = models.BooleanField(default=False)
    generic_title = models.CharField(max_length=200, blank=True, null=True)
    expert_fee_currency = models.CharField(max_length=10, blank=True, null=True)
    hourly_fee = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    language1 = models.CharField(max_length=50, blank=True, null=True)
    sector1 = models.CharField(max_length=100, blank=True, null=True)
    status = models.CharField(max_length=50, blank=True, null=True)
    skills = models.TextField(blank=True, null=True)
    products = models.TextField(blank=True, null=True)
    companies = models.TextField(blank=True, null=True)
    position = models.CharField(max_length=200, blank=True, null=True)
    seniority = models.CharField(max_length=50, blank=True, null=True)
    creation_date = models.DateTimeField(blank=True, null=True)
    last_modified = models.DateTimeField(blank=True, null=True)
    projects = models.TextField(blank=True, null=True)
    employment_history = models.TextField(blank=True, null=True)
    education = models.TextField(blank=True, null=True)
    start_date_preference = models.CharField(max_length=100, blank=True, null=True)
    work_mode = models.CharField(max_length=50, blank=True, null=True)
    availability_type = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        db_table = 'experts'
        verbose_name = 'Expert'
        verbose_name_plural = 'Experts'

    def __str__(self):
        return f"{self.first_name} {self.surname} ({self.uid})"

    def get_full_name(self):
        return f"{self.first_name} {self.surname}"

    def get_searchable_text(self):
        """Combine relevant fields for embedding generation."""
        parts = [
            self.generic_title or '',
            self.biography or '',
            self.internal_biography or '',
            self.skills or '',
            self.products or '',
            self.companies or '',
            self.position or '',
            self.projects or '',
            self.employment_history or '',
            self.education or '',
            self.sector1 or '',
        ]
        return ' '.join(filter(None, parts))
