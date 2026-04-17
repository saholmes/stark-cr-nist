use core::fmt;
use serde::{Serialize, Deserialize, Serializer, Deserializer};
use crate::traits::StarkDigest;

// ──────────────────── Digest32 (SHA3-256, BLAKE3) ────────────────────

#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest32(pub [u8; 32]);

impl StarkDigest for Digest32 {
    const SIZE: usize = 32;

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 32, "Digest32 requires exactly 32 bytes");
        let mut d = Self::default();
        d.0.copy_from_slice(bytes);
        d
    }
}

impl AsRef<[u8]> for Digest32 {
    fn as_ref(&self) -> &[u8] { &self.0 }
}

impl AsMut<[u8]> for Digest32 {
    fn as_mut(&mut self) -> &mut [u8] { &mut self.0 }
}

impl fmt::Debug for Digest32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest32({})", hex_prefix(&self.0))
    }
}

impl Serialize for Digest32 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for Digest32 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        if bytes.len() != 32 {
            return Err(serde::de::Error::custom("expected 32 bytes for Digest32"));
        }
        Ok(Digest32::from_bytes(&bytes))
    }
}

// ──────────────────── Digest48 (SHA3-384) ────────────────────

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest48(pub [u8; 48]);

impl Default for Digest48 {
    fn default() -> Self {
        Digest48([0u8; 48])
    }
}

impl StarkDigest for Digest48 {
    const SIZE: usize = 48;

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 48, "Digest48 requires exactly 48 bytes");
        let mut d = Self::default();
        d.0.copy_from_slice(bytes);
        d
    }
}

impl AsRef<[u8]> for Digest48 {
    fn as_ref(&self) -> &[u8] { &self.0 }
}

impl AsMut<[u8]> for Digest48 {
    fn as_mut(&mut self) -> &mut [u8] { &mut self.0 }
}

impl fmt::Debug for Digest48 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest48({})", hex_prefix(&self.0))
    }
}

impl Serialize for Digest48 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for Digest48 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        if bytes.len() != 48 {
            return Err(serde::de::Error::custom("expected 48 bytes for Digest48"));
        }
        Ok(Digest48::from_bytes(&bytes))
    }
}

// ──────────────────── Digest64 (SHA3-512) ────────────────────

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest64(pub [u8; 64]);

impl Default for Digest64 {
    fn default() -> Self {
        Digest64([0u8; 64])
    }
}

impl StarkDigest for Digest64 {
    const SIZE: usize = 64;

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 64, "Digest64 requires exactly 64 bytes");
        let mut d = Self::default();
        d.0.copy_from_slice(bytes);
        d
    }
}

impl AsRef<[u8]> for Digest64 {
    fn as_ref(&self) -> &[u8] { &self.0 }
}

impl AsMut<[u8]> for Digest64 {
    fn as_mut(&mut self) -> &mut [u8] { &mut self.0 }
}

impl fmt::Debug for Digest64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Digest64({})", hex_prefix(&self.0))
    }
}

impl Serialize for Digest64 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for Digest64 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        if bytes.len() != 64 {
            return Err(serde::de::Error::custom("expected 64 bytes for Digest64"));
        }
        Ok(Digest64::from_bytes(&bytes))
    }
}

// ──────────────────── Helper ────────────────────

fn hex_prefix(bytes: &[u8]) -> String {
    let hex: String = bytes.iter().take(4).map(|b| format!("{:02x}", b)).collect();
    format!("{}...", hex)
}