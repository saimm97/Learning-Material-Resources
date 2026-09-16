# Senior PHP/Symfony Interview Cheatsheet

Practice order: type each snippet from memory, don't copy-paste. If you can write it without looking, you can explain it live.

---

## 1. SOLID — one-liners to say out loud

- **S**ingle Responsibility — one reason to change.
- **O**pen/Closed — open for extension, closed for modification.
- **L**iskov Substitution — subclass must work anywhere the parent is expected.
- **I**nterface Segregation — many small interfaces over one fat one.
- **D**ependency Inversion — depend on abstractions, not concretions.

---

## 2. Strategy pattern (type this out)

```php
interface DiscountStrategy {
    public function apply(float $price): float;
}

class StudentDiscount implements DiscountStrategy {
    public function apply(float $price): float {
        return $price * 0.9;
    }
}

class SeniorDiscount implements DiscountStrategy {
    public function apply(float $price): float {
        return $price * 0.8;
    }
}

class CheckoutService {
    public function __construct(private DiscountStrategy $strategy) {}

    public function total(float $price): float {
        return $this->strategy->apply($price);
    }
}

$checkout = new CheckoutService(new StudentDiscount());
echo $checkout->total(100); // 90
```
**Say:** "Open/Closed in practice — new discount type means a new class, zero changes to CheckoutService."

---

## 3. Observer pattern (type this out)

```php
interface Observer {
    public function update(string $event): void;
}

class EmailNotifier implements Observer {
    public function update(string $event): void {
        echo "Email sent for: $event\n";
    }
}

class OrderSubject {
    private array $observers = [];

    public function attach(Observer $observer): void {
        $this->observers[] = $observer;
    }

    public function notify(string $event): void {
        foreach ($this->observers as $observer) {
            $observer->update($event);
        }
    }
}

$order = new OrderSubject();
$order->attach(new EmailNotifier());
$order->notify('order_placed');
```
**Say:** "This is literally Symfony's EventDispatcher — dispatch() = notify(), listeners = observers."

---

## 4. Decorator pattern (type this out)

```php
interface Coffee {
    public function cost(): float;
}

class SimpleCoffee implements Coffee {
    public function cost(): float { return 2.0; }
}

class MilkDecorator implements Coffee {
    public function __construct(private Coffee $coffee) {}
    public function cost(): float { return $this->coffee->cost() + 0.5; }
}

$coffee = new MilkDecorator(new SimpleCoffee());
echo $coffee->cost(); // 2.5
```
**Say:** "Symfony's `decorates:` service tag does exactly this — wraps an existing service without touching its class."

---

## 5. Factory pattern (type this out)

```php
interface Exporter {
    public function export(): string;
}

class PdfExporter implements Exporter {
    public function export(): string { return "PDF"; }
}

class CsvExporter implements Exporter {
    public function export(): string { return "CSV"; }
}

class ExporterFactory {
    public static function create(string $type): Exporter {
        return match ($type) {
            'pdf' => new PdfExporter(),
            'csv' => new CsvExporter(),
            default => throw new InvalidArgumentException("Unknown type: $type"),
        };
    }
}

$exporter = ExporterFactory::create('pdf');
echo $exporter->export();
```

---

## 6. SPL interfaces (type all 4, they WILL ask)

```php
// ArrayAccess
class Settings implements ArrayAccess {
    private array $data = [];
    public function offsetExists($key): bool { return isset($this->data[$key]); }
    public function offsetGet($key): mixed { return $this->data[$key] ?? null; }
    public function offsetSet($key, $value): void { $this->data[$key] = $value; }
    public function offsetUnset($key): void { unset($this->data[$key]); }
}

// Countable
class Cart implements Countable {
    private array $items = [];
    public function add(string $item): void { $this->items[] = $item; }
    public function count(): int { return count($this->items); }
}

// IteratorAggregate (preferred over Iterator — less code)
class UserCollection implements IteratorAggregate {
    private array $users = [];
    public function add(string $u): void { $this->users[] = $u; }
    public function getIterator(): Iterator { return new ArrayIterator($this->users); }
}
```
**Say:** "Symfony's ParameterBag, HeaderBag, and Doctrine's ArrayCollection implement these so they behave like arrays."

---

## 7. Closures — value vs reference (type this)

```php
$x = 10;
$byValue = function () use ($x) { return $x; };
$byRef   = function () use (&$x) { return $x; };

$x = 99;

echo $byValue(); // 10
echo $byRef();   // 99

// arrow fn always captures by value automatically
$fn = fn() => $x + 1;
```

---

## 8. Doctrine toIterable() (say this from memory)

```php
$query = $em->createQuery('SELECT u FROM App\Entity\User u');

foreach ($query->toIterable() as $user) {
    // process one row at a time — memory stays flat
    // even for millions of rows
}
```
**Say:** "findAll() loads everything into memory. toIterable() streams rows one at a time, so it scales to millions of records without exhausting memory."

---

## 9. TDD — red/green/refactor (write ONE real test)

```php
use PHPUnit\Framework\TestCase;

class CalculatorTest extends TestCase
{
    public function testAddsTwoNumbers(): void
    {
        $calculator = new Calculator();
        $this->assertEquals(5, $calculator->add(2, 3));
    }
}

class Calculator
{
    public function add(int $a, int $b): int
    {
        return $a + $b;
    }
}
```
**Say:** "Red — write a failing test. Green — write minimum code to pass. Refactor — clean up without breaking the test."

---

## 10. DTO vs Entity vs Repository (type this mini flow)

```php
// Entity — tied to the DB via Doctrine
#[ORM\Entity]
class User {
    #[ORM\Id, ORM\GeneratedValue, ORM\Column]
    private int $id;
    #[ORM\Column]
    private string $email;
    // getters/setters
}

// Repository — talks to DB, returns Entities
class UserRepository extends ServiceEntityRepository {
    public function findActiveUsers(): array {
        return $this->createQueryBuilder('u')
            ->where('u.active = true')
            ->getQuery()
            ->getResult();
    }
}

// DTO — no logic, just a data shape for moving data across layers
final class UserDTO {
    public function __construct(
        public readonly string $email,
        public readonly string $displayName,
    ) {}
}

// Mapping Entity -> DTO before returning as API response
function toDTO(User $user): UserDTO {
    return new UserDTO(
        email: $user->getEmail(),
        displayName: $user->getDisplayName(),
    );
}
```
**Say:** "Entity is the domain object tied to the DB. Repository isolates data access and speaks in Entities. DTO is a lightweight, logic-free shape used to move data between layers without leaking internal Entity structure."

---

## 11. Symfony DI — the one they'll actually watch you write

```php
interface PaymentGateway {
    public function charge(float $amount): bool;
}

class StripeGateway implements PaymentGateway {
    public function charge(float $amount): bool {
        // call Stripe API
        return true;
    }
}

class CheckoutService {
    // depends on the ABSTRACTION, not StripeGateway directly
    public function __construct(private PaymentGateway $gateway) {}

    public function pay(float $amount): bool {
        return $this->gateway->charge($amount);
    }
}
```
```yaml
# config/services.yaml
services:
    App\Service\PaymentGateway: '@App\Service\StripeGateway'
```
**Say:** "Symfony's container makes constructor injection and interface binding easy — but it doesn't automatically make code SOLID. If I type-hint StripeGateway directly instead of the interface, autowiring still works, I've just glued myself to Stripe anyway."

---

## 12. Symfony vs Ruby on Rails — equivalent concepts

| Symfony | Ruby on Rails equivalent | What it does |
|---|---|---|
| Entity (Doctrine) | Model (ActiveRecord) | The class tied to a database table, with an ORM handling persistence. |
| Repository | ActiveRecord query methods / Scopes | Rails blends this into the Model itself (`User.active`) rather than a separate Repository class. |
| DTO | Form Object / PORO (Plain Old Ruby Object) | A plain, logic-light object for moving structured data across layers without exposing the Model directly. |
| Voter (Security) | Policy (via the Pundit gem) | Per-object authorization: can THIS user do THIS action on THIS specific record? |
| EventSubscriber / Event | ActiveSupport::Notifications or Callbacks (`after_create`, etc.) | Reacting to something happening, decoupled from the code that triggered it. |
| Validator (custom constraint) | Custom Validator class (`ActiveModel::Validator`) | Same exact concept — Rails has this built in too. |
| Form Type | Form Object (often a plain Ruby class including `ActiveModel::Model`) | Rails has no dedicated Form component — people build their own Form Objects for the same purpose. |
| Messenger (async jobs/queues) | Active Job (with Sidekiq/Resque as backend) | Background job dispatching — `dispatch_later` vs `perform_later`. |
| Notifier | ActionMailer + notification gems (e.g. Noticed) | Sending emails/SMS/push through a unified interface. |
| Command (console) | Rake task | `php bin/console app:something` vs a custom Rake task. |
| DataFixtures | Seeds (`db/seeds.rb`) or Factory Bot | Populating fake/sample data for dev and tests. |
| Service class | Service Object | Same concept, same name — business logic that doesn't belong on the Model or Controller. |
| Serializer component | ActiveModel::Serializer or Jbuilder | Converting objects to/from JSON for API responses. |
| Dependency Injection Container | No direct equivalent — Rails uses Ruby's own object instantiation, sometimes the dry-container gem | Rails leans on convention over an explicit DI container. |
| Attributes (`#[Route]`, `#[Assert\...]`) | DSL macros (`validates`, `resources :books`) | Same idea — metadata expressed inline on the class — PHP attributes vs Ruby's DSL style. |

**Say if it comes up:** "A lot of these patterns aren't Symfony-specific — Voters map to Pundit Policies, Messenger maps to Active Job, Service classes are Service Objects in both worlds. The concepts are the same; only the naming and how explicit the framework is about it differs. Symfony tends to be more explicit and DI-heavy, Rails leans more on convention."

---

## Final 10-minute drill before you walk in

1. Say all 5 SOLID principles out loud, one sentence each — no notes.
2. Type the Strategy pattern from memory, no looking.
3. Type one PHPUnit test from memory.
4. Say the one-line difference: DTO vs Entity vs Repository vs DAO.
5. Say what toIterable() solves and why it matters at scale.

If you can do all 5 without hesitation, you're ready. Good luck.
