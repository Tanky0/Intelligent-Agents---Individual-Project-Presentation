"""Uses the operating system's certificate store to verify HTTPS connections."""
# On the computer used to develop and test the application, antivirus software
# inspects HTTPS traffic and replaces each website's certificate with one signed
# by its own root certificate. Windows trusts this certificate, but Python's
# bundled certificate list does not. As a result, requests to Gemini, NCBI and
# Crossref failed certificate verification.
#
# Using the operating system's certificate store keeps certificate verification
# enabled while recognising certificates already trusted by the computer.
# Disabling antivirus HTTPS inspection would reduce protection. Adding the root
# certificate to Python's bundled list would apply only to one computer and
# could be overwritten during an upgrade. This approach allows each computer to
# use its own trusted certificates when running the submitted code (R-01).

# Reads the certificates the computer already trusts. Added only to solve the
# problem described above.
import truststore


def install() -> None:
    """Tells Python to check certificates against the computer's own list."""
    # Calling this more than once makes no difference.
    truststore.inject_into_ssl()
