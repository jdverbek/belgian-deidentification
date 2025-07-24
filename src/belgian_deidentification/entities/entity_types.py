"""Entity type definitions for the Belgian deidentification system."""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import re


class EntityType(str, Enum):
    """Types of entities that can be detected and deidentified."""
    PERSON = "PERSON"
    ADDRESS = "ADDRESS"
    DATE = "DATE"
    MEDICAL_ID = "MEDICAL_ID"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    AGE = "AGE"
    OTHER = "OTHER"


class EntitySubtype(str, Enum):
    """Subtypes for more specific entity classification."""
    # Person subtypes
    FIRST_NAME = "FIRST_NAME"
    LAST_NAME = "LAST_NAME"
    FULL_NAME = "FULL_NAME"
    INITIALS = "INITIALS"
    
    # Address subtypes
    STREET_ADDRESS = "STREET_ADDRESS"
    POSTAL_CODE = "POSTAL_CODE"
    CITY = "CITY"
    COUNTRY = "COUNTRY"
    
    # Date subtypes
    BIRTH_DATE = "BIRTH_DATE"
    ADMISSION_DATE = "ADMISSION_DATE"
    DISCHARGE_DATE = "DISCHARGE_DATE"
    APPOINTMENT_DATE = "APPOINTMENT_DATE"
    
    # Medical ID subtypes
    PATIENT_ID = "PATIENT_ID"
    MEDICAL_RECORD_NUMBER = "MEDICAL_RECORD_NUMBER"
    INSURANCE_NUMBER = "INSURANCE_NUMBER"
    SOCIAL_SECURITY_NUMBER = "SOCIAL_SECURITY_NUMBER"
    
    # Organization subtypes
    HOSPITAL = "HOSPITAL"
    CLINIC = "CLINIC"
    DEPARTMENT = "DEPARTMENT"
    INSURANCE_COMPANY = "INSURANCE_COMPANY"


@dataclass
class Entity:
    """Base entity class for detected entities."""
    text: str
    start: int
    end: int
    entity_type: EntityType
    subtype: Optional[EntitySubtype] = None
    confidence: float = 0.0
    source: str = "unknown"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def span(self) -> tuple:
        """Get the span as a tuple."""
        return (self.start, self.end)
    
    @property
    def length(self) -> int:
        """Get the length of the entity text."""
        return len(self.text)
    
    def overlaps_with(self, other: 'Entity') -> bool:
        """Check if this entity overlaps with another entity."""
        return not (self.end <= other.start or other.end <= self.start)
    
    def contains(self, other: 'Entity') -> bool:
        """Check if this entity contains another entity."""
        return self.start <= other.start and self.end >= other.end
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'entity_type': self.entity_type.value,
            'subtype': self.subtype.value if self.subtype else None,
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary."""
        return cls(
            text=data['text'],
            start=data['start'],
            end=data['end'],
            entity_type=EntityType(data['entity_type']),
            subtype=EntitySubtype(data['subtype']) if data.get('subtype') else None,
            confidence=data.get('confidence', 0.0),
            source=data.get('source', 'unknown'),
            metadata=data.get('metadata', {}),
        )


@dataclass
class PHIEntity(Entity):
    """PHI-specific entity with additional privacy-related metadata."""
    risk_level: str = "medium"  # low, medium, high, critical
    replacement_text: Optional[str] = None
    action_taken: Optional[str] = None  # removed, replaced, masked
    validation_status: Optional[str] = None  # validated, needs_review, rejected
    
    def __post_init__(self):
        super().__post_init__()
        self.metadata.update({
            'is_phi': True,
            'risk_level': self.risk_level,
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PHI entity to dictionary."""
        data = super().to_dict()
        data.update({
            'risk_level': self.risk_level,
            'replacement_text': self.replacement_text,
            'action_taken': self.action_taken,
            'validation_status': self.validation_status,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PHIEntity':
        """Create PHI entity from dictionary."""
        return cls(
            text=data['text'],
            start=data['start'],
            end=data['end'],
            entity_type=EntityType(data['entity_type']),
            subtype=EntitySubtype(data['subtype']) if data.get('subtype') else None,
            confidence=data.get('confidence', 0.0),
            source=data.get('source', 'unknown'),
            metadata=data.get('metadata', {}),
            risk_level=data.get('risk_level', 'medium'),
            replacement_text=data.get('replacement_text'),
            action_taken=data.get('action_taken'),
            validation_status=data.get('validation_status'),
        )


class EntityPatterns:
    """Predefined patterns for entity recognition."""
    
    # Dutch name patterns
    DUTCH_TUSSENVOEGSELS = [
        'van', 'de', 'der', 'den', 'van der', 'van den', 'van de',
        'ter', 'ten', 'te', 'op', 'aan', 'bij', 'in', 'onder',
        'over', 'voor', 'uit', 'tot', 'met', 'zonder'
    ]
    
    # Belgian postal code pattern (4 digits)
    BELGIAN_POSTAL_CODE = re.compile(r'\b[1-9]\d{3}\b')
    
    # Dutch phone number patterns
    DUTCH_PHONE_PATTERNS = [
        re.compile(r'\b0[1-9]\d{1,2}[-\s]?\d{6,7}\b'),  # Landline
        re.compile(r'\b06[-\s]?\d{8}\b'),  # Mobile
        re.compile(r'\+31[-\s]?[1-9]\d{1,2}[-\s]?\d{6,7}\b'),  # International
    ]
    
    # Email pattern
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    # Date patterns (Dutch formats)
    DATE_PATTERNS = [
        re.compile(r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}\b'),  # DD-MM-YYYY
        re.compile(r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2}\b'),   # DD-MM-YY
        re.compile(r'\b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b'),   # YYYY-MM-DD
    ]
    
    # Medical ID patterns
    MEDICAL_ID_PATTERNS = [
        re.compile(r'\b\d{8,12}\b'),  # General numeric IDs
        re.compile(r'\b[A-Z]{2,3}\d{6,10}\b'),  # Alphanumeric IDs
        re.compile(r'\bPAT\d{6,8}\b', re.IGNORECASE),  # Patient IDs
    ]
    
    # Age patterns
    AGE_PATTERNS = [
        re.compile(r'\b\d{1,3}[-\s]?jaar(?:s)?\b', re.IGNORECASE),
        re.compile(r'\b\d{1,3}[-\s]?jr\b', re.IGNORECASE),
        re.compile(r'\bleeftijd:?\s*\d{1,3}\b', re.IGNORECASE),
    ]
    
    # Belgian city names (common ones)
    BELGIAN_CITIES = [
        'Brussel', 'Antwerpen', 'Gent', 'Charleroi', 'Luik', 'Brugge',
        'Namur', 'Leuven', 'Mons', 'Aalst', 'Mechelen', 'La Louvière',
        'Kortrijk', 'Hasselt', 'Sint-Niklaas', 'Oostende', 'Tournai',
        'Genk', 'Seraing', 'Roeselare', 'Mouscron', 'Verviers', 'Dendermonde',
        'Beringen', 'Turnhout', 'Vilvoorde', 'Lokeren', 'Sint-Truiden'
    ]
    
    @classmethod
    def get_name_pattern(cls) -> re.Pattern:
        """Get pattern for Dutch names with tussenvoegsels."""
        tussenvoegsel_pattern = '|'.join(re.escape(t) for t in cls.DUTCH_TUSSENVOEGSELS)
        return re.compile(
            rf'\b[A-Z][a-z]+(?:\s+(?:{tussenvoegsel_pattern})\s+[A-Z][a-z]+)*\b'
        )
    
    @classmethod
    def get_address_pattern(cls) -> re.Pattern:
        """Get pattern for Belgian addresses."""
        return re.compile(
            r'\b[A-Z][a-z]+(?:straat|laan|plein|weg|steenweg|boulevard|avenue|rue|place)\s+\d+[a-z]?\b',
            re.IGNORECASE
        )
    
    @classmethod
    def is_dutch_name_part(cls, text: str) -> bool:
        """Check if text could be part of a Dutch name."""
        text_lower = text.lower()
        
        # Check if it's a tussenvoegsel
        if text_lower in cls.DUTCH_TUSSENVOEGSELS:
            return True
        
        # Check if it's a capitalized word (potential name)
        if text[0].isupper() and text[1:].islower() and text.isalpha():
            return True
        
        return False
    
    @classmethod
    def is_belgian_postal_code(cls, text: str) -> bool:
        """Check if text is a Belgian postal code."""
        return bool(cls.BELGIAN_POSTAL_CODE.match(text))
    
    @classmethod
    def is_dutch_phone_number(cls, text: str) -> bool:
        """Check if text is a Dutch phone number."""
        return any(pattern.match(text) for pattern in cls.DUTCH_PHONE_PATTERNS)
    
    @classmethod
    def is_email(cls, text: str) -> bool:
        """Check if text is an email address."""
        return bool(cls.EMAIL_PATTERN.match(text))
    
    @classmethod
    def is_date(cls, text: str) -> bool:
        """Check if text is a date."""
        return any(pattern.match(text) for pattern in cls.DATE_PATTERNS)
    
    @classmethod
    def is_medical_id(cls, text: str) -> bool:
        """Check if text is a medical ID."""
        return any(pattern.match(text) for pattern in cls.MEDICAL_ID_PATTERNS)
    
    @classmethod
    def is_age(cls, text: str) -> bool:
        """Check if text represents an age."""
        return any(pattern.search(text) for pattern in cls.AGE_PATTERNS)
    
    @classmethod
    def is_belgian_city(cls, text: str) -> bool:
        """Check if text is a Belgian city name."""
        return text in cls.BELGIAN_CITIES


class EntityValidator:
    """Validator for entity detection results."""
    
    @staticmethod
    def validate_person_name(text: str) -> bool:
        """Validate if text is likely a person name."""
        # Must be alphabetic with possible spaces and hyphens
        if not re.match(r'^[A-Za-z\s\-\']+$', text):
            return False
        
        # Must start with capital letter
        if not text[0].isupper():
            return False
        
        # Check for common non-name patterns
        non_name_patterns = [
            r'^(de|het|een|dit|dat|deze|die)\s',  # Articles
            r'^(en|of|maar|want|dus)\s',  # Conjunctions
            r'^(in|op|aan|bij|van|voor)\s',  # Prepositions
        ]
        
        text_lower = text.lower()
        for pattern in non_name_patterns:
            if re.match(pattern, text_lower):
                return False
        
        return True
    
    @staticmethod
    def validate_address(text: str) -> bool:
        """Validate if text is likely an address."""
        # Must contain street indicators and numbers
        street_indicators = [
            'straat', 'laan', 'plein', 'weg', 'steenweg', 'boulevard',
            'avenue', 'rue', 'place', 'square'
        ]
        
        text_lower = text.lower()
        has_street_indicator = any(indicator in text_lower for indicator in street_indicators)
        has_number = bool(re.search(r'\d+', text))
        
        return has_street_indicator and has_number
    
    @staticmethod
    def validate_date(text: str) -> bool:
        """Validate if text is a valid date."""
        # Check basic date format
        if not EntityPatterns.is_date(text):
            return False
        
        # Additional validation could include:
        # - Valid day/month ranges
        # - Reasonable year ranges
        # - Leap year validation
        
        return True
    
    @staticmethod
    def validate_medical_id(text: str) -> bool:
        """Validate if text is likely a medical ID."""
        # Must be alphanumeric
        if not re.match(r'^[A-Za-z0-9]+$', text):
            return False
        
        # Must have reasonable length
        if len(text) < 4 or len(text) > 20:
            return False
        
        # Must contain at least one digit
        if not any(c.isdigit() for c in text):
            return False
        
        return True
    
    @staticmethod
    def validate_phone(text: str) -> bool:
        """Validate if text is a valid phone number."""
        return EntityPatterns.is_dutch_phone_number(text)
    
    @staticmethod
    def validate_email(text: str) -> bool:
        """Validate if text is a valid email."""
        return EntityPatterns.is_email(text)
    
    @classmethod
    def validate_entity(cls, entity: Entity) -> bool:
        """Validate an entity based on its type."""
        validators = {
            EntityType.PERSON: cls.validate_person_name,
            EntityType.ADDRESS: cls.validate_address,
            EntityType.DATE: cls.validate_date,
            EntityType.MEDICAL_ID: cls.validate_medical_id,
            EntityType.PHONE: cls.validate_phone,
            EntityType.EMAIL: cls.validate_email,
        }
        
        validator = validators.get(entity.entity_type)
        if validator:
            return validator(entity.text)
        
        return True  # No specific validator, assume valid

